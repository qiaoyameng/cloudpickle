"""Tests for the protocol 5 out-of-band buffer bundle API.

These tests only rely on the standard library (``pickle.PickleBuffer`` and
``memoryview``); no NumPy dependency is required.
"""

import gc
import weakref

import pytest

from pickle import PickleBuffer

import cloudpickle
from cloudpickle import (
    BufferChunk,
    BufferIntegrityError,
    BufferManifestEntry,
    BufferOrderError,
    BundleSizeLimitError,
    MissingBufferError,
    PickleBundle,
    WritableZeroCopyError,
    dumps_bundle,
    loads_bundle,
)

pytestmark = pytest.mark.skipif(
    cloudpickle.DEFAULT_PROTOCOL < 5,
    reason="out-of-band buffers require pickle protocol 5",
)


class BufferObject:
    """Object whose payload goes through one protocol 5 out-of-band buffer.

    The payload is wrapped in a ``PickleBuffer`` at dump time.  Depending on
    the buffers handed to the unpickler, the reconstructed payload is either
    a fresh ``bytearray``/view (copy mode) or the original ``PickleBuffer``
    (zero-copy mode).
    """

    def __init__(self, data):
        self._buffer = PickleBuffer(data)

    def __reduce_ex__(self, protocol):
        return (BufferObject._reconstruct, (self._buffer,))

    @staticmethod
    def _reconstruct(buffer):
        obj = BufferObject.__new__(BufferObject)
        obj.buffer = buffer
        return obj


class Segment:
    """Weak-referenceable stand-in for independently stored chunk storage."""

    def __init__(self, data, readonly):
        self.data = bytes(data) if readonly else bytearray(data)

    def __buffer__(self, flags):
        return memoryview(self.data)


def _store(bundle):
    return {chunk.id: chunk.tobytes() for chunk in bundle}


def _segment_getter(store, lent):
    def get_buffer(entry):
        raw = store[entry.id]
        seg = Segment(raw, entry.readonly)
        lent[entry.id] = weakref.ref(seg)
        return memoryview(seg)

    return get_buffer


def test_bundle_shape_and_small_main_stream():
    ro = BufferObject(b"readonly-payload")
    rw = BufferObject(bytearray(b"writable-payload"))
    bundle = dumps_bundle({"ro": ro, "rw": rw, "meta": [1, 2, 3]})

    assert isinstance(bundle, PickleBundle)
    assert len(bundle) == 2
    assert [entry.id for entry in bundle.manifest] == ["0", "1"]
    assert [entry.order for entry in bundle.manifest] == [0, 1]
    assert [entry.readonly for entry in bundle.manifest] == [True, False]
    for entry in bundle.manifest:
        assert entry.length > 0
        assert len(entry.digest) == 64  # sha256 hex digest

    # The large payloads live in the chunks, not in the main pickle stream.
    assert b"readonly-payload" not in bundle.stream
    assert b"writable-payload" not in bundle.stream

    chunks = list(bundle)
    assert all(isinstance(chunk, BufferChunk) for chunk in chunks)
    assert chunks[0].readonly is True
    assert isinstance(chunks[0].tobytes(), bytes)


def test_copy_mode_roundtrip_restores_writable_copies():
    rw = BufferObject(bytearray(b"writable-payload"))
    bundle = dumps_bundle({"rw": rw})
    store = _store(bundle)

    back = loads_bundle(bundle.stream, bundle.manifest, lambda e: store[e.id])
    assert isinstance(back["rw"].buffer, bytearray)
    back["rw"].buffer[0] = ord("W")
    assert back["rw"].buffer.startswith(b"W")


def test_dumps_bundle_with_low_protocol_is_upgraded():
    bundle = dumps_bundle(BufferObject(b"data"), protocol=2)
    assert bundle.stream[1] == 5


def test_alias_same_storage_emits_separate_entries_isolated_in_copy_mode():
    # Protocol 5 emits one buffer per *occurrence*; one underlying storage
    # referenced twice yields two manifest entries.
    shared = bytearray(b"shared-underlying-storage")
    bundle = dumps_bundle([BufferObject(shared), BufferObject(shared)])
    assert len(bundle) == 2
    assert bundle.manifest[0].digest == bundle.manifest[1].digest
    assert bundle.manifest[0].id != bundle.manifest[1].id

    store = _store(bundle)
    first, second = loads_bundle(
        bundle.stream, bundle.manifest, lambda e: store[e.id]
    )
    assert first.buffer is not second.buffer
    first.buffer[:6] = b"XXXXXX"
    assert bytes(second.buffer[:6]) == b"shared"


def test_equal_independent_writable_buffers_are_not_deduplicated():
    bundle = dumps_bundle(
        [
            BufferObject(bytearray(b"equal-content")),
            BufferObject(bytearray(b"equal-content")),
        ]
    )
    assert len(bundle) == 2
    store = _store(bundle)

    first, second = loads_bundle(
        bundle.stream, bundle.manifest, lambda e: store[e.id]
    )
    first.buffer[0] = ord("?")
    assert second.buffer[0] == ord("e")


def test_missing_buffer_none_and_keyerror():
    bundle = dumps_bundle(
        [BufferObject(b"a"), BufferObject(b"bb")]
    )
    store = _store(bundle)

    incomplete = dict(store)
    del incomplete["1"]
    with pytest.raises(MissingBufferError, match="1"):
        loads_bundle(
            bundle.stream, bundle.manifest, lambda e: incomplete.get(e.id)
        )

    def raise_keyerror(entry):
        raise KeyError(entry.id)

    with pytest.raises(MissingBufferError, match="0"):
        loads_bundle(bundle.stream, bundle.manifest, raise_keyerror)


def test_manifest_order_and_duplicate_ids_checked_before_fetching():
    bundle = dumps_bundle([BufferObject(b"a"), BufferObject(b"bb")])
    entries = list(bundle.manifest)

    fetched = []

    def get_buffer(entry):
        fetched.append(entry.id)
        raise AssertionError("must not be called before order validation")

    with pytest.raises(BufferOrderError):
        loads_bundle(bundle.stream, [entries[1], entries[0]], get_buffer)
    assert fetched == []

    bad = [
        entries[0],
        BufferManifestEntry(
            id="0",  # duplicated id
            order=1,
            length=2,
            digest="a" * 64,
            readonly=True,
        ),
    ]
    with pytest.raises(BufferOrderError):
        loads_bundle(bundle.stream, bad, get_buffer)


def test_tampered_length_and_digest_detected():
    bundle = dumps_bundle(BufferObject(b"integrity"))
    entry = bundle.manifest[0]

    wrong_length = {entry.id: b"x" * (entry.length - 1)}
    with pytest.raises(BufferIntegrityError, match="expected"):
        loads_bundle(bundle.stream, bundle.manifest, lambda e: wrong_length[e.id])

    wrong_content = {entry.id: b"x" * entry.length}
    with pytest.raises(BufferIntegrityError, match="digest mismatch"):
        loads_bundle(bundle.stream, bundle.manifest, lambda e: wrong_content[e.id])


def test_total_size_cap_enforced_before_fetching():
    bundle = dumps_bundle(
        [BufferObject(b"a" * 100), BufferObject(b"b" * 50)]
    )

    def fail_get(entry):
        raise AssertionError("no chunk should be fetched under the cap")

    with pytest.raises(BundleSizeLimitError):
        loads_bundle(
            bundle.stream,
            bundle.manifest,
            fail_get,
            max_total_bytes=99,
        )

    # The cap can be disabled entirely.
    store = _store(bundle)
    loads_bundle(
        bundle.stream, bundle.manifest, lambda e: store[e.id],
        max_total_bytes=None,
    )


def test_zero_copy_readonly_lifetime_held_and_released():
    bundle = dumps_bundle(BufferObject(b"zero-copy-readonly"))
    store = _store(bundle)
    lent = {}
    get_buffer = _segment_getter(store, lent)

    holder = loads_bundle(
        bundle.stream, bundle.manifest, get_buffer, zero_copy=True
    )
    graph = holder.obj
    assert lent["0"]() is not None
    with graph.buffer.raw() as mv:
        assert mv.readonly
        assert bytes(mv) == b"zero-copy-readonly"

    holder.close()
    gc.collect()
    assert lent["0"]() is None


def test_zero_copy_context_manager_releases_on_exit():
    bundle = dumps_bundle(BufferObject(b"zero-copy-readonly"))
    store = _store(bundle)
    lent = {}
    get_buffer = _segment_getter(store, lent)

    with loads_bundle(
        bundle.stream, bundle.manifest, get_buffer, zero_copy=True
    ) as graph:
        with graph.buffer.raw() as mv:
            assert bytes(mv) == b"zero-copy-readonly"
        assert lent["0"]() is not None

    gc.collect()
    assert lent["0"]() is None


def test_zero_copy_writable_rejects_readonly_storage():
    bundle = dumps_bundle(BufferObject(bytearray(b"writable")))
    store = _store(bundle)

    with pytest.raises(WritableZeroCopyError, match="writable"):
        loads_bundle(
            bundle.stream,
            bundle.manifest,
            lambda e: store[e.id],  # bytes -> read-only
            zero_copy=True,
        )


def test_zero_copy_writable_writes_reach_borrowed_storage():
    bundle = dumps_bundle(BufferObject(bytearray(b"writable-buffer")))
    store = _store(bundle)
    segment = bytearray(store["0"])

    with loads_bundle(
        bundle.stream,
        bundle.manifest,
        lambda e: PickleBuffer(segment),
        zero_copy=True,
    ) as graph:
        with graph.buffer.raw() as mv:
            assert not mv.readonly
            mv[0] = ord("Z")

    assert segment[0] == ord("Z")


def test_zero_copy_alias_preserved_when_getter_aliases_storage():
    shared = bytearray(b"shared-underlying-storage")
    bundle = dumps_bundle([BufferObject(shared), BufferObject(shared)])
    aliased = PickleBuffer(shared)

    with loads_bundle(
        bundle.stream,
        bundle.manifest,
        lambda e: aliased,  # same borrowed object for both ids
        zero_copy=True,
    ) as (first, second):
        assert first.buffer is second.buffer
        with first.buffer.raw() as mv:
            mv[0] = ord("#")
        with second.buffer.raw() as mv:
            assert mv[0] == ord("#")


def test_zero_copy_distinct_storage_stays_distinct():
    shared = bytearray(b"shared-underlying-storage")
    bundle = dumps_bundle([BufferObject(shared), BufferObject(shared)])

    s1 = bytearray(shared)
    s2 = bytearray(shared)
    with loads_bundle(
        bundle.stream,
        bundle.manifest,
        lambda e: PickleBuffer(s1 if e.id == "0" else s2),
        zero_copy=True,
    ) as (first, second):
        assert first.buffer is not second.buffer
        with first.buffer.raw() as mv:
            mv[0] = ord("Q")
        with second.buffer.raw() as mv:
            assert mv[0] == ord("s")


def test_fetch_failure_releases_borrowed_resources_copy_mode():
    bundle = dumps_bundle(
        [BufferObject(b"first"), BufferObject(bytearray(b"second"))]
    )
    store = _store(bundle)
    lent = {}

    def get_buffer(entry):
        if entry.id == "1":
            raise KeyError(entry.id)
        seg = Segment(store[entry.id], entry.readonly)
        lent[entry.id] = weakref.ref(seg)
        return memoryview(seg)

    with pytest.raises(MissingBufferError):
        loads_bundle(bundle.stream, bundle.manifest, get_buffer)

    gc.collect()
    assert lent["0"]() is None


def test_fetch_failure_releases_borrowed_resources_zero_copy():
    bundle = dumps_bundle(
        [BufferObject(b"first"), BufferObject(bytearray(b"second"))]
    )
    store = _store(bundle)
    lent = {}

    def get_buffer(entry):
        if entry.id == "1":
            raise KeyError(entry.id)
        seg = Segment(store[entry.id], entry.readonly)
        lent[entry.id] = weakref.ref(seg)
        return memoryview(seg)

    with pytest.raises(MissingBufferError):
        loads_bundle(
            bundle.stream, bundle.manifest, get_buffer, zero_copy=True
        )

    gc.collect()
    assert lent["0"]() is None


def test_zero_copy_tampering_detected_without_touching_pickle():
    bundle = dumps_bundle(BufferObject(b"integrity"))
    with pytest.raises(BufferIntegrityError, match="digest"):
        loads_bundle(
            bundle.stream,
            bundle.manifest,
            lambda e: b"x" * bundle.manifest[0].length,
            zero_copy=True,
        )


def test_plain_dumps_loads_unchanged():
    obj = {"answer": 42, "fn": lambda x: x + 1}
    assert cloudpickle.loads(cloudpickle.dumps(obj))["answer"] == 42

    # Classic protocol 5 buffer_callback / buffers roundtrip is untouched.
    emitted = []
    stream = cloudpickle.dumps(
        BufferObject(b"classic-p5"),
        protocol=5,
        buffer_callback=emitted.append,
    )
    incoming = []
    for pb in emitted:
        with pb.raw() as mv:
            incoming.append(bytes(mv))
    back = cloudpickle.loads(stream, buffers=incoming)
    assert bytes(back.buffer) == b"classic-p5"


def test_dynamic_cloudpickle_content_survives_bundle():
    make_adder = lambda n: (lambda x: x + n)  # noqa: E731
    obj = {"adder": make_adder(10), "big": BufferObject(b"side-buffer")}
    bundle = dumps_bundle(obj)
    store = _store(bundle)

    back = loads_bundle(bundle.stream, bundle.manifest, lambda e: store[e.id])
    assert back["adder"](32) == 42
