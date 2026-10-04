"""Protocol 5 out-of-band buffer "bundle" API.

The regular :func:`cloudpickle.dumps` / :func:`cloudpickle.loads` functions
embed protocol 5 out-of-band buffers directly into the pickle stream when no
``buffer_callback`` is given.  This module implements an alternative layout for
objects that carry *large* buffers (typically produced by libraries such as
NumPy): the large payloads are stored separately while the main pickle only
carries a small object graph.

A bundle is made of three parts:

* a ``main`` pickled stream, as :class:`bytes`;
* a :class:`BufferManifest` describing each out-of-band buffer: its position
  (order), length, cryptographic digest and the read-only constraint;
* the raw buffer contents themselves, which can be iterated over at dumping
  time and fetched individually by id at loading time.

The manifest never contains the buffer data: only metadata and a digest used
to detect tampering.  Buffer contents are addressed by an opaque ``chunk_id``.

Alias semantics
---------------
Protocol 5 distinguishes two situations and this module preserves that
distinction on the receiving side:

* When the *same underlying buffer* is referenced from several places (either
  the very same :class:`pickle.PickleBuffer` object appears several times in
  the object graph, or distinct ``PickleBuffer`` objects expose the same
  exporting object), the manifest groups those references into a single chunk
  (same ``chunk_id``) and :func:`load_bundle` hands out views of the very same
  memory for every reference: the alias is preserved.

* Two originally independent, writable buffers that merely contain *equal*
  bytes are never merged (no content based de-duplication): they keep distinct
  chunk ids and distinct storage, so writing through one restored view can
  never mutate the other.

Only the identity of the underlying exporting object is used for grouping,
never the bytes' content.

Trust and validation
--------------------
All manifest checks (missing chunks, order mismatch, digest mismatch, total
size limit, read-only constraints) are performed *before* the pickle stream is
given to the (trusted) pickle machinery.  No arbitrary code is executed by
:func:`load_bundle` before the manifest has been fully validated.

Zero copy
---------
When ``copy=False`` (the default), the restored object keeps memoryviews
pointing at the buffers returned by the caller-supplied ``get_chunk``
function.  The storage behind those buffers must stay alive and valid until the
restored object is done with them.  :meth:`LoadedObject.release` releases only
memoryviews that the loader itself created (for instance read-only aliases
produced from a writable source); it never releases a memoryview handed back
directly by ``get_chunk``, whose lifetime remains owned by the caller.

This module only relies on the standard library (:mod:`pickle`,
:class:`pickle.PickleBuffer`, :class:`memoryview`); NumPy is not required.
"""

import hashlib
import hmac
import io
import json
import pickle
import threading
import uuid

from .cloudpickle import Pickler

__all__ = [
    "dump_bundle",
    "dumps_bundle",
    "load_bundle",
    "BufferEntry",
    "BufferManifest",
    "BufferBundle",
    "LoadedObject",
    "BundleError",
    "ManifestError",
    "MissingChunkError",
    "ChunkOrderError",
    "BufferTamperedError",
    "BufferLimitError",
    "BufferReadOnlyError",
    "DEFAULT_MAX_TOTAL_SIZE",
    "MANIFEST_FORMAT",
]

#: Protocol identifier of the manifest serialization format.
MANIFEST_FORMAT = "cloudpickle-buffer-bundle/1"

#: Default upper bound (in bytes) for the sum of all buffer lengths.  This is
#: applied regardless of the caller-supplied storage size so that a corrupt or
#: malicious manifest cannot trick the loader into an unbounded allocation.
#: ``None`` disables the limit.
DEFAULT_MAX_TOTAL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB

# A single flat, C-contiguous unsigned-byte view is the canonical representation
# used for digesting and for the bytes that back the storage ids.
_CANONICAL_FORMAT = "B"
_DIGEST_ALGORITHM = "sha256"
_DIGEST_PREFIX = "sha256:"
_HASH_BLOCK_SIZE = 1024 * 1024


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BundleError(Exception):
    """Base class for all bundle related errors."""


class ManifestError(BundleError):
    """The manifest itself is malformed or internally inconsistent."""


class MissingChunkError(BundleError):
    """A chunk referenced by the manifest could not be retrieved."""


class ChunkOrderError(BundleError):
    """The manifest entries are not ordered like the pickle expects."""


class BufferTamperedError(BundleError):
    """A retrieved chunk's length or digest does not match the manifest."""


class BufferLimitError(BundleError):
    """The total buffer size declared by the manifest exceeds the limit."""


class BufferReadOnlyError(BundleError):
    """A chunk declared writable is only available as a read-only view."""


# --------------------------------------------------------------------------- #
# Manifest data model
# --------------------------------------------------------------------------- #


class BufferEntry:
    """Metadata for a single out-of-band buffer occurrence.

    Each ``PickleBuffer`` seen while pickling produces one entry, in emission
    order.  Entries are addressed through :attr:`chunk_id`; several entries may
    share the same chunk id to encode aliasing.

    Attributes:
        index: Zero based position of the entry, equal to its position in the
            list of protocol 5 buffer opcodes.
        chunk_id: Opaque identifier used to fetch the buffer content.
        length: Number of bytes of the flat buffer.
        digest: ``"sha256:"`` prefixed hex digest of the bytes.
        readonly: When true the buffer must be exposed read-only; when false
            the restored object expects a writable buffer.
    """

    __slots__ = ("index", "chunk_id", "length", "digest", "readonly")

    def __init__(self, index, chunk_id, length, digest, readonly):
        self.index = int(index)
        self.chunk_id = chunk_id
        self.length = int(length)
        self.digest = digest
        self.readonly = bool(readonly)

    def to_dict(self):
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "length": self.length,
            "digest": self.digest,
            "readonly": self.readonly,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                index=data["index"],
                chunk_id=data["chunk_id"],
                length=data["length"],
                digest=data["digest"],
                readonly=data["readonly"],
            )
        except (KeyError, TypeError) as exc:
            raise ManifestError(f"Invalid buffer entry: {data!r}") from exc

    def __eq__(self, other):
        if not isinstance(other, BufferEntry):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        mode = "ro" if self.readonly else "rw"
        return (
            f"BufferEntry(index={self.index}, chunk_id={self.chunk_id!r}, "
            f"length={self.length}, {mode})"
        )


class BufferManifest:
    """Ordered list of :class:`BufferEntry` with the format identifier."""

    def __init__(self, entries=(), format=MANIFEST_FORMAT):
        self.format = format
        self.entries = list(entries)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    @property
    def total_length(self):
        return sum(entry.length for entry in self.entries)

    @property
    def chunk_ids(self):
        """Chunk ids in first-occurrence order (unique storage objects)."""
        seen = set()
        ordered = []
        for entry in self.entries:
            if entry.chunk_id not in seen:
                seen.add(entry.chunk_id)
                ordered.append(entry.chunk_id)
        return ordered

    def to_json(self, indent=None):
        """Serialize the manifest (metadata only, no buffer data) to JSON."""
        payload = {
            "format": self.format,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        return json.dumps(payload, indent=indent).encode("utf-8")

    @classmethod
    def from_json(cls, raw):
        """Parse JSON produced by :meth:`to_json`."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ManifestError("Manifest is not valid JSON") from exc
        if not isinstance(payload, dict) or "entries" not in payload:
            raise ManifestError("Manifest is missing the 'entries' list")
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise ManifestError("Manifest 'entries' must be a list")
        return cls(
            entries=[BufferEntry.from_dict(item) for item in entries],
            format=payload.get("format", MANIFEST_FORMAT),
        )

    def __eq__(self, other):
        if not isinstance(other, BufferManifest):
            return NotImplemented
        return (
            self.format == other.format
            and self.entries == other.entries
        )

    def __repr__(self):
        return f"BufferManifest({self.format!r}, {len(self)} entries)"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _canonical_view(view):
    """Return a flat, C-contiguous unsigned-byte view of ``view``.

    Raises :class:`ManifestError`-like buffer errors when the buffer cannot be
    expressed in that canonical shape.
    """
    if not isinstance(view, memoryview):
        raise BufferTamperedError(
            f"Expected a memoryview-compatible chunk, got {type(view).__name__}"
        )
    try:
        # Cast to a flat unsigned-byte view without changing the read-only
        # status: a writable source must stay writable (and vice versa).
        return view.cast("B")
    except (TypeError, ValueError) as exc:
        raise BufferTamperedError(
            "Chunk cannot be represented as a flat unsigned-byte buffer"
        ) from exc


def _digest_view(view):
    """Stream-hash a canonical flat memoryview (avoids large copies)."""
    hasher = hashlib.sha256()
    update = hasher.update
    step = _HASH_BLOCK_SIZE
    length = len(view)
    start = 0
    while start < length:
        update(view[start : start + step])
        start += step
    return _DIGEST_PREFIX + hasher.hexdigest()


# --------------------------------------------------------------------------- #
# Dumping
# --------------------------------------------------------------------------- #


class _BundleCollector:
    """Buffer callback and bookkeeping used while building a bundle."""

    def __init__(self):
        self.pickle_buffers = []
        # (id(base), readonly) -> chunk_id, keeps alias groups together.
        self._chunks = {}
        self._base_refs = {}
        self._lock = threading.Lock()
        self.entries = []

    def __call__(self, pickle_buffer):
        raw = pickle_buffer.raw()
        base = raw.obj
        readonly = bool(raw.readonly)
        key = (id(base), readonly)
        with self._lock:
            self.pickle_buffers.append(pickle_buffer)
            chunk_id = self._chunks.get(key)
            if chunk_id is None:
                chunk_id = uuid.uuid4().hex
                self._chunks[key] = chunk_id
                # Keep a strong reference to the exporting object for the whole
                # duration of the pickling/iteration so that the id cannot be
                # recycled while the bundle is in use.
                self._base_refs[key] = base
            index = len(self.entries)
            digest = _digest_view(raw)
            self.entries.append(
                BufferEntry(
                    index=index,
                    chunk_id=chunk_id,
                    length=len(raw),
                    digest=digest,
                    readonly=readonly,
                )
            )


class BufferBundle:
    """Result of :func:`dump_bundle`.

    Attributes:
        main: The main pickled stream (small object graph, no buffer data).
        manifest: The :class:`BufferManifest` describing every buffer.
    """

    def __init__(self, main, collector):
        self.main = main
        self.manifest = BufferManifest(collector.entries)
        self._pickle_buffers = collector.pickle_buffers
        # Keep the exporting objects alive for the lifetime of the bundle.
        self._base_refs = collector._base_refs
        self._closed = False
        # chunk_id -> memoryview for unique storage objects.
        self._chunk_views = {}
        for entry, pickle_buffer in zip(
            self.manifest.entries, self._pickle_buffers
        ):
            if entry.chunk_id not in self._chunk_views:
                self._chunk_views[entry.chunk_id] = pickle_buffer.raw()

    # -- iteration --------------------------------------------------------- #

    @property
    def manifest_bytes(self):
        """JSON-encoded manifest (metadata only)."""
        return self.manifest.to_json()

    def chunks(self):
        """Yield ``(chunk_id, memoryview)`` once per unique chunk.

        The yielded memoryviews are zero-copy views over the original object
        and stay valid while the bundle is open (or used as a context manager).
        Iteration order matches the first occurrence of each chunk in the
        manifest.
        """
        if self._closed:
            raise BundleError("Cannot iterate over a closed bundle")
        for chunk_id in self.manifest.chunk_ids:
            yield chunk_id, self._chunk_views[chunk_id]

    def __iter__(self):
        return self.chunks()

    # -- lifetime ---------------------------------------------------------- #

    def close(self):
        """Release the underlying PickleBuffer objects.

        After closing, memoryviews handed out by the bundle must not be used
        anymore.
        """
        if self._closed:
            return
        self._closed = True
        for view in self._chunk_views.values():
            try:
                view.release()
            except (ValueError, TypeError):
                pass
        for pickle_buffer in self._pickle_buffers:
            try:
                pickle_buffer.release()
            except (ValueError, TypeError):
                pass
        self._chunk_views = {}
        self._base_refs = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"BufferBundle(main={len(self.main)} bytes, "
            f"{len(self.manifest)} buffer entries, "
            f"{len(self.manifest.chunk_ids)} chunks)"
        )


def dumps_bundle(obj, protocol=None):
    """Serialize ``obj`` and return an open :class:`BufferBundle`.

    The bundle must be closed (either explicitly with
    :meth:`BufferBundle.close` or through a ``with`` block) once the buffer
    contents have been read.

    ``protocol`` must support protocol 5 out-of-band buffers (``protocol >= 5``)
    and defaults to 5.  Using a lower protocol raises :class:`ValueError`.
    """
    if protocol is None:
        protocol = 5
    protocol = int(protocol)
    if protocol < 5:
        raise ValueError(
            "dumps_bundle requires pickle protocol 5 or later to emit "
            "out-of-band buffers"
        )
    collector = _BundleCollector()
    with io.BytesIO() as stream:
        pickler = Pickler(stream, protocol=protocol, buffer_callback=collector)
        pickler.dump(obj)
        main = stream.getvalue()
    return BufferBundle(main, collector)


def dump_bundle(obj, protocol=None):
    """Serialize ``obj`` to a protocol 5 buffer bundle.

    Returns a :class:`BufferBundle` exposing the main stream, the manifest and
    an iterable of unique ``(chunk_id, memoryview)`` buffer contents.  See
    :func:`dumps_bundle` for details.
    """
    return dumps_bundle(obj, protocol=protocol)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class LoadedObject:
    """Holder for a restored object plus its zero-copy buffer lifetimes.

    The restored object is available as :attr:`obj`.  In zero-copy mode the
    returned object may reference memoryviews backed by the chunks returned by
    ``get_chunk``; those borrows stay valid until :meth:`release` is called or
    the object is used as a context manager::

        with load_bundle(main, manifest, storage.get) as loaded:
            use(loaded.obj)
        # borrowed buffers are released here

    When loaded with ``copy=True``, the object owns independent writable copies
    of its buffers and releasing the holder earlier is harmless (but still
    supported for symmetry).
    """

    def __init__(self, obj, owned_views=(), borrowed_views=()):
        self.obj = obj
        self._owned_views = list(owned_views)
        self._borrowed_views = list(borrowed_views)
        self._released = False

    def release(self):
        """Release memoryviews borrowed for the zero-copy restoration.

        Only views created by :func:`load_bundle` itself are released; the
        caller-supplied storage objects are not touched.
        """
        if self._released:
            return
        self._released = True
        for view in self._owned_views:
            try:
                view.release()
            except (ValueError, TypeError):
                pass
        # Borrowed views are exactly the memoryviews backing ``obj``; release
        # them after the caller is done with the restored object.
        for view in self._borrowed_views:
            try:
                view.release()
            except (ValueError, TypeError):
                pass

    close = release

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _validate_manifest_structure(manifest, max_total_size):
    """Static checks performed before any chunk is fetched."""
    if not isinstance(manifest, BufferManifest):
        raise ManifestError("manifest must be a BufferManifest instance")
    entries = manifest.entries
    for position, entry in enumerate(entries):
        if not isinstance(entry, BufferEntry):
            raise ManifestError(f"Entry {position} is not a BufferEntry")
        if entry.index != position:
            raise ChunkOrderError(
                f"Entry at position {position} declares index {entry.index}; "
                "manifest is missing or reorders buffers"
            )
        if not isinstance(entry.chunk_id, str) or not entry.chunk_id:
            raise ManifestError(f"Entry {position} has an invalid chunk id")
        if entry.length < 0:
            raise ManifestError(f"Entry {position} has a negative length")
        if not isinstance(entry.digest, str) or not entry.digest.startswith(
            _DIGEST_PREFIX
        ):
            raise ManifestError(
                f"Entry {position} digest must use {_DIGEST_PREFIX!r}"
            )

    # Entries sharing a chunk id describe the same storage; their immutable
    # metadata must agree (length/digest/readonly), otherwise the manifest is
    # internally contradictory.
    grouped = {}
    for entry in entries:
        previous = grouped.get(entry.chunk_id)
        if previous is None:
            grouped[entry.chunk_id] = entry
        else:
            if (
                previous.length != entry.length
                or previous.digest != entry.digest
                or previous.readonly != entry.readonly
            ):
                raise ManifestError(
                    f"Entries sharing chunk id {entry.chunk_id!r} disagree on "
                    "length/digest/readonly"
                )

    if max_total_size is not None and manifest.total_length > max_total_size:
        raise BufferLimitError(
            f"Declared buffer size {manifest.total_length} bytes exceeds the "
            f"limit of {max_total_size} bytes"
        )


def load_bundle(
    main,
    manifest,
    get_chunk,
    copy=False,
    max_total_size=DEFAULT_MAX_TOTAL_SIZE,
):
    """Restore an object dumped with :func:`dumps_bundle`.

    Parameters:
        main: The main pickled stream returned by :func:`dumps_bundle`.
        manifest: The matching :class:`BufferManifest` (or JSON bytes/str).
        get_chunk: Callable taking a chunk id and returning the chunk content
            as a bytes-like object or :class:`memoryview`.  Returning ``None``
            (or raising :class:`KeyError`) signals a missing chunk.
        copy: When true, materialize independent writable copies of every
            chunk; the result does not borrow any memory from ``get_chunk``.
            When false (default), restore zero-copy and keep borrows alive via
            the returned :class:`LoadedObject`.
        max_total_size: Upper bound on the total buffer length.

    Every chunk is validated for presence, order, length, digest and the
    read-only constraint *before* the pickle stream is executed.  If fetching
    a chunk fails, all already borrowed resources are released.
    """
    if isinstance(manifest, (bytes, bytearray, str)):
        manifest = BufferManifest.from_json(manifest)

    _validate_manifest_structure(manifest, max_total_size)

    # chunk_id -> canonical view, fetched only once even if aliased.
    unique_views = {}
    # Per-entry views, in opcode order.
    buffer_views = []
    # Views that must be released on failure / on release of the result.
    owned_views = []
    borrowed_views = []
    running_total = 0

    def _release_all():
        for view in owned_views:
            try:
                view.release()
            except (ValueError, TypeError):
                pass
        for view in borrowed_views:
            try:
                view.release()
            except (ValueError, TypeError):
                pass

    try:
        for entry in manifest.entries:
            chosen = unique_views.get(entry.chunk_id)
            if chosen is not None:
                # Alias: reuse the exact same view so references stay shared.
                buffer_views.append(chosen)
                continue

            chunk = None
            try:
                chunk = get_chunk(entry.chunk_id)
            except KeyError as exc:
                raise MissingChunkError(
                    f"No chunk for id {entry.chunk_id!r}"
                ) from exc
            except MissingChunkError:
                raise
            except Exception as exc:
                raise MissingChunkError(
                    f"Failed to fetch chunk {entry.chunk_id!r}: {exc}"
                ) from exc
            if chunk is None:
                raise MissingChunkError(f"No chunk for id {entry.chunk_id!r}")

            try:
                fetched = memoryview(chunk)
            except TypeError as exc:
                raise BufferTamperedError(
                    f"Chunk {entry.chunk_id!r} is not a buffer"
                ) from exc

            canonical = _canonical_view(fetched)

            # Length check: guards against a truncated / oversized chunk.
            if len(canonical) != entry.length:
                raise BufferTamperedError(
                    f"Chunk {entry.chunk_id!r} has length {len(canonical)}, "
                    f"expected {entry.length}"
                )

            # Digest check: detects any byte-level tampering.
            if not hmac.compare_digest(_digest_view(canonical), entry.digest):
                raise BufferTamperedError(
                    f"Digest mismatch for chunk {entry.chunk_id!r}: buffer "
                    "content has been tampered with"
                )

            if copy:
                # One independent copy per unique chunk; aliases below share it.
                chosen = memoryview(bytearray(canonical))
                owned_views.append(chosen)
            else:
                if entry.readonly:
                    if not canonical.readonly:
                        # Expose a read-only alias without copying.  This is a
                        # view created by the loader itself, so it is tracked
                        # for release; views handed straight back from
                        # ``get_chunk`` remain owned by the caller.
                        canonical = canonical.toreadonly()
                        borrowed_views.append(canonical)
                elif canonical.readonly:
                    # The pickle expects a writable buffer, which the opcodes
                    # do not enforce on every interpreter; fail up front.
                    raise BufferReadOnlyError(
                        f"Chunk {entry.chunk_id!r} is declared writable but "
                        "only a read-only view is available; pass copy=True to "
                        "materialize a writable copy"
                    )
                chosen = canonical

            # Back the manifest-level limit with the actual fetched total.
            running_total += entry.length
            if max_total_size is not None and running_total > max_total_size:
                raise BufferLimitError(
                    f"Fetched buffer total exceeds the limit of "
                    f"{max_total_size} bytes"
                )

            unique_views[entry.chunk_id] = chosen
            buffer_views.append(chosen)

        # All manifest checks passed: only now run the trusted pickle restore.
        try:
            obj = pickle.loads(main, buffers=iter(buffer_views))
        except pickle.UnpicklingError as exc:
            if "buffer" in str(exc):
                raise ChunkOrderError(
                    "Main stream and manifest do not agree on the number or "
                    f"order of buffers: {exc}"
                ) from exc
            raise
    except BaseException:
        _release_all()
        raise

    if copy:
        # Copies own their storage; nothing borrowed keeps ``get_chunk`` alive.
        return LoadedObject(obj, owned_views=(), borrowed_views=())

    return LoadedObject(
        obj,
        owned_views=owned_views,
        borrowed_views=borrowed_views,
    )
