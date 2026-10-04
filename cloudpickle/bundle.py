"""Out-of-band buffer bundle API (pickle protocol 5).

This module layers an explicit buffer manifest on top of protocol 5's
out-of-band buffer machinery (PEP 574).  The pickle stream emitted by
:func:`dumps_bundle` only carries small, in-band data; large buffers are
externalized as an ordered list of self-describing :class:`BufferChunk`
objects that the caller may store independently (object stores, shared
memory, ...).  :func:`loads_bundle` reconstructs objects by asking the
caller to fetch each chunk's bytes through an ``id``-keyed callback.

The manifest records, for every buffer, its emission order, length, a
SHA-256 digest and whether the pickle stream declared it read-only.
Before any (untrusted) pickle code runs, the loader verifies that no chunk
is missing, that chunks arrive in the exact emission order, that no chunk
was tampered with and that the total buffered data stays below a
caller-controlled limit.

Aliasing semantics
------------------
Protocol 5 emits one buffer *per occurrence* in the object graph, even
when the very same ``PickleBuffer`` is referenced several times; pickle
itself performs no content-based deduplication.  :func:`dumps_bundle`
follows the same rule:

* Equal manifest entries mean "two independent emissions", not "one shared
  chunk".  Buffers that merely contain equal bytes but were independent
  writable buffers at dump time keep independent entries and are restored
  on independent writable storage, so content-based sharing can never
  couple writes of unrelated objects.
* True aliasing (one underlying storage referenced from several places)
  shows up as repeated manifest entries.  In copy mode every occurrence is
  restored from its own fetched copy, so writes stay isolated; in
  zero-copy mode aliasing is preserved exactly when the caller-provided
  ``get_buffer`` returns the same borrowed storage for the corresponding
  ids (for example by mapping several aliased ids to one shared-memory
  segment).
"""

from dataclasses import dataclass, field
import hashlib
import io
import pickle
from typing import Any, Callable, Iterable, Iterator, List, Optional

from .cloudpickle import DEFAULT_PROTOCOL, Pickler

# Protocol 5 is the first protocol supporting out-of-band buffers.
BUNDLE_PROTOCOL = 5

# Default safety bound for the aggregated length of all externalized
# buffers.  Can be overridden per ``loads_bundle`` call; ``None`` disables
# the check entirely.
DEFAULT_MAX_TOTAL_BYTES = 4 * 1024 ** 3  # 4 GiB


class BundleError(Exception):
    """Base class for bundle loading errors."""


class MissingBufferError(BundleError):
    """A manifest entry has no corresponding chunk in the storage backend."""


class BufferOrderError(BundleError):
    """Manifest entries are out of order or duplicated."""


class BufferIntegrityError(BundleError):
    """A fetched chunk does not match its manifest length or digest."""


class BundleSizeLimitError(BundleError):
    """The manifest advertises more bytes than the configured limit."""


class WritableZeroCopyError(BundleError):
    """Zero-copy restore of a writable buffer was not possible."""


@dataclass(frozen=True)
class BufferManifestEntry:
    """Description of one out-of-band buffer emission.

    Attributes
    ----------
    id:
        Stable identifier of the chunk; it is simply the emission index
        (as a string) and is independent of the chunk's content, so
        identical buffers never collide.
    order:
        Zero-based emission order, matching the order in which protocol 5
        consumes the ``buffers`` argument.
    length:
        Size of the buffer in bytes.
    digest:
        Hexadecimal SHA-256 digest of the buffer contents.
    readonly:
        ``True`` when the pickle stream marked the buffer read-only (the
        ``READONLY_BUFFER`` opcode).  Writable buffers must be restored on
        mutable storage; read-only buffers may be shared freely.
    format, ndim:
        Original memoryview metadata, recorded for information only.
    """

    id: str
    order: int
    length: int
    digest: str
    readonly: bool
    format: str = "B"
    ndim: int = 1


@dataclass
class BufferChunk:
    """A manifest entry together with the actual buffer contents."""

    entry: BufferManifestEntry
    buffer: pickle.PickleBuffer

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def readonly(self) -> bool:
        return self.entry.readonly

    def tobytes(self) -> bytes:
        """Return an independent, in-memory ``bytes`` copy of the chunk."""
        with self.buffer.raw() as mv:
            if mv.c_contiguous:
                return bytes(mv)
            return mv.tobytes()


@dataclass
class PickleBundle:
    """Small pickle stream plus the externalized buffer chunks.

    ``chunks`` is ordered exactly like the protocol 5 ``buffers``
    argument.  Iterating over the bundle yields its chunks, so they can be
    streamed to independent storage without materializing all of them in
    memory at once.
    """

    stream: bytes
    chunks: List[BufferChunk] = field(default_factory=list)

    @property
    def manifest(self) -> List[BufferManifestEntry]:
        return [chunk.entry for chunk in self.chunks]

    def __iter__(self) -> Iterator[BufferChunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


def _buffer_digest(mv: memoryview) -> str:
    h = hashlib.sha256()
    if mv.c_contiguous:
        h.update(mv)
    else:
        h.update(mv.tobytes())
    return h.hexdigest()


def _make_chunk(order: int, pb: pickle.PickleBuffer) -> BufferChunk:
    with pb.raw() as mv:
        entry = BufferManifestEntry(
            id=str(order),
            order=order,
            length=mv.nbytes,
            digest=_buffer_digest(mv),
            readonly=bool(mv.readonly),
            format=mv.format,
            ndim=mv.ndim,
        )
        # Keep the bundle self-contained and independent of the caller's
        # storage: snapshot each emission into its own bytes object.
        # Equal but originally independent buffers keep distinct storage;
        # there is deliberately no content-based deduplication.
        data = bytes(mv) if mv.c_contiguous else mv.tobytes()
    return BufferChunk(entry=entry, buffer=pickle.PickleBuffer(data))


def dumps_bundle(obj: Any, protocol: Optional[int] = None) -> PickleBundle:
    """Serialize ``obj`` into a small pickle stream and external buffers.

    Every protocol 5 out-of-band buffer emitted while pickling ``obj`` is
    collected into its own :class:`BufferChunk`.  The returned stream does
    not embed any of the buffers' contents and can be stored or sent
    independently of the chunks.

    Protocols below 5 cannot carry out-of-band buffers, so they are
    upgraded to 5.
    """
    if protocol is None:
        protocol = DEFAULT_PROTOCOL
    if protocol < BUNDLE_PROTOCOL:
        protocol = BUNDLE_PROTOCOL

    raw_buffers: List[pickle.PickleBuffer] = []
    with io.BytesIO() as file:
        Pickler(
            file,
            protocol=protocol,
            buffer_callback=raw_buffers.append,
        ).dump(obj)
        stream = file.getvalue()

    chunks = [_make_chunk(i, pb) for i, pb in enumerate(raw_buffers)]
    return PickleBundle(stream=stream, chunks=chunks)


def _normalize_manifest(
    manifest: Iterable[BufferManifestEntry],
) -> List[BufferManifestEntry]:
    entries = list(manifest)
    seen_ids = set()
    normalized: List[BufferManifestEntry] = []
    for expected_order, entry in enumerate(entries):
        if entry.order != expected_order:
            raise BufferOrderError(
                f"manifest entry {entry.id!r} declares order {entry.order}, "
                f"expected {expected_order}"
            )
        if entry.id in seen_ids:
            raise BufferOrderError(
                f"duplicate buffer id in manifest: {entry.id!r}"
            )
        seen_ids.add(entry.id)
        if entry.length < 0:
            raise BufferIntegrityError(
                f"negative length in manifest entry {entry.id!r}"
            )
        if len(entry.digest) != 64:
            raise BufferIntegrityError(
                f"invalid sha256 digest in manifest entry {entry.id!r}"
            )
        normalized.append(entry)
    return normalized


# A get_buffer callback returns *borrowed* storage for a chunk id.  ``bytes``
# inputs are read-only; ``bytearray`` and writable ``memoryview`` are not.
# The loader is responsible for releasing what it borrowed when loading
# fails (copy mode) or when the zero-copy holder is closed.
GetBuffer = Callable[[BufferManifestEntry], Any]


def _release(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if close is None:
        return
    try:
        close()
    except ValueError:
        # Already released.
        pass


class _ZeroCopyResult:
    """Holder keeping borrowed zero-copy buffers alive.

    Each borrowed chunk is handed to the unpickler as a ``PickleBuffer``,
    and protocol 5 passes that same object through to the reconstructed
    graph (or a read-only ``memoryview`` of it for objects that request
    one).  Either way the graph points straight into the storage returned
    by ``get_buffer``; that storage must stay alive and un-released until
    the caller is done with the graph.
    """

    def __init__(self, obj: Any, pbs: List[pickle.PickleBuffer]):
        self._obj = obj
        self._pbs = pbs
        self._closed = False

    @property
    def obj(self) -> Any:
        return self._obj

    def __enter__(self) -> Any:
        return self._obj

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pb in self._pbs:
            pb.release()
        self._pbs = []


class _CountingIterator:
    """Iterate ``incoming`` while counting how many buffers were consumed.

    The C implementation of ``pickle.Unpickler`` does not expose the buffer
    queue, so consumption is tracked here to detect stream/manifest mismatch.
    """

    def __init__(self, items):
        self._iter = iter(items)
        self.consumed = 0
        self.remaining = 0
        for _ in items:
            self.remaining += 1

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self._iter)
        self.consumed += 1
        self.remaining -= 1
        return value


def _unpickle(stream: bytes, incoming: List[Any]) -> Any:
    buffers = _CountingIterator(incoming)
    obj = pickle.Unpickler(io.BytesIO(stream), buffers=buffers).load()
    if buffers.remaining:
        # More buffers were advertised than the stream consumes: this
        # indicates a manifest/stream mismatch.
        raise BundleError("fewer buffers consumed than listed in the manifest")
    return obj


def loads_bundle(
    stream: bytes,
    manifest: Iterable[BufferManifestEntry],
    get_buffer: GetBuffer,
    *,
    max_total_bytes: Optional[int] = DEFAULT_MAX_TOTAL_BYTES,
    zero_copy: bool = False,
):
    """Load a pickle bundle produced by :func:`dumps_bundle`.

    Parameters
    ----------
    stream:
        The small pickle stream (``PickleBundle.stream``).
    manifest:
        Ordered chunk descriptions (``PickleBundle.manifest``).  They are
        checked for order, duplicate ids and digest shape before any chunk
        is fetched.
    get_buffer:
        Callable invoked once per entry, in manifest order, returning the
        chunk's contents as ``bytes``, ``bytearray``, ``memoryview`` or
        ``pickle.PickleBuffer``.  Raising ``KeyError``/``FileNotFoundError``
        or returning ``None`` reports a missing chunk.
    max_total_bytes:
        Hard cap on the aggregated advertised length of the buffers.  It is
        enforced *before* fetching anything, so a hostile manifest cannot
        trigger unbounded storage allocation.  ``None`` disables it.
    zero_copy:
        When ``False`` (default) every chunk is copied into fresh storage
        owned by the reconstructed objects; borrowed resources are released
        before this function returns, and distinct entries are restored as
        independently writable copies.

        When ``True`` chunks are handed to the unpickler without copying.
        The returned :class:`_ZeroCopyResult` owns the lifetime of every
        borrowed resource; keep it (or use it as a context manager) alive
        for as long as the reconstructed object graph may reference the
        buffers.  Read-only storage for an entry declared writable raises
        :class:`WritableZeroCopyError` instead of silently copying.

    All manifest checks (missing chunks, ordering, length, digest and the
    total-size cap) run before the trusted pickle restoration itself; any
    failure releases the resources already borrowed.
    """
    entries = _normalize_manifest(manifest)

    total = sum(entry.length for entry in entries)
    if max_total_bytes is not None and total > max_total_bytes:
        raise BundleSizeLimitError(
            f"bundle advertises {total} buffer bytes, over limit "
            f"{max_total_bytes}"
        )

    incoming: List[Any] = []
    # PickleBuffers handed to the unpickler in zero-copy mode; they (and the
    # caller storage they wrap) must outlive this call.
    pbs: List[pickle.PickleBuffer] = []
    # Borrowed resources that need explicit release if loading fails.
    pending_views: List[memoryview] = []

    def _abort():
        for pending in pending_views:
            _release(pending)
        pending_views.clear()
        for pb in pbs:
            pb.release()
        pbs.clear()

    try:
        for entry in entries:
            try:
                borrowed = get_buffer(entry)
            except (KeyError, FileNotFoundError) as exc:
                _abort()
                raise MissingBufferError(
                    f"no chunk available for buffer {entry.id!r}"
                ) from exc
            if borrowed is None:
                _abort()
                raise MissingBufferError(
                    f"no chunk available for buffer {entry.id!r}"
                )

            if zero_copy:
                # A true zero-copy handoff requires PickleBuffer instances in
                # the buffers argument.  Register the PickleBuffer before
                # validating so that _abort() uniformly releases it on any
                # failure (including a PickleBuffer handed in by the caller,
                # since no object graph escapes a failed load).
                pb = (
                    borrowed
                    if isinstance(borrowed, pickle.PickleBuffer)
                    else pickle.PickleBuffer(borrowed)
                )
                pbs.append(pb)
                # The raw view is used for validation only; the context
                # manager releases it immediately, so it never pins borrowed
                # storage beyond this check.
                with pb.raw() as mv:
                    if not entry.readonly and mv.readonly:
                        raise WritableZeroCopyError(
                            f"buffer {entry.id!r} is declared writable but "
                            "get_buffer returned read-only storage"
                        )
                    if mv.nbytes != entry.length:
                        raise BufferIntegrityError(
                            f"buffer {entry.id!r}: expected {entry.length} "
                            f"bytes, got {mv.nbytes}"
                        )
                    if _buffer_digest(mv) != entry.digest:
                        raise BufferIntegrityError(
                            f"buffer {entry.id!r}: digest mismatch "
                            "(contents tampered with?)"
                        )
            else:
                mv = memoryview(borrowed)
                pending_views.append(mv)
                try:
                    if mv.nbytes != entry.length:
                        raise BufferIntegrityError(
                            f"buffer {entry.id!r}: expected {entry.length} "
                            f"bytes, got {mv.nbytes}"
                        )
                    if _buffer_digest(mv) != entry.digest:
                        raise BufferIntegrityError(
                            f"buffer {entry.id!r}: digest mismatch "
                            "(contents tampered with?)"
                        )
                    # One fresh, owned copy per occurrence: read-only slots
                    # get immutable bytes, writable slots get an independent
                    # bytearray, so aliased or equal entries never share
                    # writable storage through deduplication.
                    incoming.append(
                        bytes(mv) if entry.readonly else bytearray(mv)
                    )
                except BaseException:
                    _abort()
                    raise
                else:
                    # The copy owns its own storage now: release the borrowed
                    # view immediately instead of pinning it until return.
                    mv.release()
                    pending_views.remove(mv)

        obj = _unpickle(stream, list(pbs) if zero_copy else incoming)
    except BaseException:
        _abort()
        raise

    if zero_copy:
        return _ZeroCopyResult(obj, pbs)
    return obj
