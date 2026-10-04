"""Tests for cloudpickle's protocol 5 out-of-band buffer bundle API.

These tests only rely on the standard library (``pickle.PickleBuffer`` and
``memoryview``); NumPy is intentionally not required.
"""

import gc
import json
import weakref

import pytest

pickle = pytest.importorskip("pickle")

from pickle import PickleBuffer  # noqa: E402

import cloudpickle  # noqa: E402
from cloudpickle import bundle  # noqa: E402


# A stand-in for remote/independent storage: bytes are copied, so views
# returned by the writable getter do not alias the original pickled objects.
class Storage:
    def __init__(self):
        self.blobs = {}

    def store(self, b):
        for chunk_id, view in b.chunks():
            self.blobs[chunk_id] = bytes(view)

    def writable(self, chunk_id):
        return bytearray(self.blobs[chunk_id])

    def readonly(self, chunk_id):
        # bytes expose read-only memoryviews
        return self.blobs[chunk_id]

    def missing(self, chunk_id):
        return None

    def keyerror(self, chunk_id):
        raise KeyError(chunk_id)


def make_bundle(obj, protocol=None):
    return cloudpickle.dump_bundle(obj, protocol=protocol)


def dump_to_storage(obj, storage, protocol=None, as_json=False):
    with make_bundle(obj, protocol=protocol) as b:
        storage.store(b)
        manifest = b.manifest.to_json() if as_json else b.manifest
        return b.main, manifest


# --------------------------------------------------------------------------- #
# Basic round trips
# --------------------------------------------------------------------------- #


def test_basic_writable_roundtrip():
    storage = Storage()
    data = bytearray(b"hello big buffer " * 100)
    obj = {"data": PickleBuffer(data), "meta": [1, 2, 3]}
    main, manifest = dump_to_storage(obj, storage)

    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        result = loaded.obj
        assert result["meta"] == [1, 2, 3]
        view = result["data"]
        assert bytes(view[:17]) == b"hello big buffer "
        assert view.readonly is False
        view[0] = ord("X")
        assert view[0] == ord("X")


def test_main_stream_is_small():
    # The big payload must not be inlined into the main pickle stream.
    storage = Storage()
    data = bytearray(b"x" * 100_000)
    main, _ = dump_to_storage(PickleBuffer(data), storage)
    assert len(main) < 1000


def test_readonly_bytes_roundtrip_with_json_manifest():
    storage = Storage()
    main, manifest_json = dump_to_storage(
        [PickleBuffer(b"read only data")], storage, as_json=True
    )
    assert isinstance(manifest_json, (bytes, bytearray))
    payload = json.loads(manifest_json.decode())
    assert payload["format"] == bundle.MANIFEST_FORMAT
    assert payload["entries"][0]["readonly"] is True

    with cloudpickle.load_bundle(main, manifest_json, storage.readonly) as loaded:
        view = loaded.obj[0]
        assert view.readonly is True
        assert bytes(view) == b"read only data"


def test_no_buffers():
    storage = Storage()
    main, manifest = dump_to_storage({"x": 1}, storage)
    assert len(manifest) == 0
    assert manifest.chunk_ids == []
    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        assert loaded.obj == {"x": 1}


def test_dynamic_function_and_buffer():
    storage = Storage()

    def make():
        scale = 3
        return lambda value: value * scale, PickleBuffer(bytearray(b"abcdef"))

    func, buf = make()
    main, manifest = dump_to_storage((func, buf), storage)
    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        restored_func, restored_buf = loaded.obj
        assert restored_func(7) == 21
        assert bytes(restored_buf) == b"abcdef"


# --------------------------------------------------------------------------- #
# Alias semantics
# --------------------------------------------------------------------------- #


def test_same_picklebuffer_preserves_alias():
    storage = Storage()
    shared = bytearray(b"SHARED")
    pb = PickleBuffer(shared)
    main, manifest = dump_to_storage([pb, pb], storage)

    # One entry per opcode occurrence, but a single underlying chunk.
    assert len(manifest.entries) == 2
    assert len(manifest.chunk_ids) == 1

    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        first, second = loaded.obj
        assert first.obj is second.obj  # alias preserved across fresh copies
        first[0] = ord("Z")
        assert bytes(second) == b"ZHARED"


def test_separate_picklebuffers_same_base_preserve_alias():
    storage = Storage()
    base = bytearray(b"SAMEBASE")
    main, manifest = dump_to_storage(
        [PickleBuffer(base), PickleBuffer(base)], storage
    )
    assert len(manifest.chunk_ids) == 1
    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        first, second = loaded.obj
        assert first.obj is second.obj


def test_equal_independent_writable_buffers_not_deduplicated():
    storage = Storage()
    left = bytearray(b"EQUAL")
    right = bytearray(b"EQUAL")
    main, manifest = dump_to_storage(
        [PickleBuffer(left), PickleBuffer(right)], storage
    )
    # Equal content but independent storage must remain distinct chunks.
    assert len(manifest.chunk_ids) == 2
    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        first, second = loaded.obj
        assert first.obj is not second.obj
        first[0] = ord("!")
        assert bytes(first) == b"!QUAL"
        assert bytes(second) == b"EQUAL"  # the other is untouched


def test_same_base_different_readonly_gets_separate_chunks():
    storage = Storage()
    base = bytearray(b"MIXED")
    main, manifest = dump_to_storage(
        [PickleBuffer(base), PickleBuffer(memoryview(base).toreadonly())],
        storage,
    )
    assert len(manifest.chunk_ids) == 2
    assert [entry.readonly for entry in manifest.entries] == [False, True]
    with cloudpickle.load_bundle(main, manifest, storage.writable) as loaded:
        writable, readonly = loaded.obj
        assert writable.readonly is False
        assert readonly.readonly is True


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #


def test_missing_chunk_none():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abc")), storage)
    with pytest.raises(bundle.MissingChunkError):
        cloudpickle.load_bundle(main, manifest, storage.missing)


def test_missing_chunk_keyerror_and_other_error():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abc")), storage)
    with pytest.raises(bundle.MissingChunkError):
        cloudpickle.load_bundle(main, manifest, storage.keyerror)

    def boom(chunk_id):
        raise RuntimeError("storage down")

    with pytest.raises(bundle.MissingChunkError):
        cloudpickle.load_bundle(main, manifest, boom)


def test_tampered_digest():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)

    def tampered(chunk_id):
        blob = bytearray(storage.blobs[chunk_id])
        blob[0] ^= 0xFF
        return blob

    with pytest.raises(bundle.BundleError):
        cloudpickle.load_bundle(main, manifest, tampered)


def test_wrong_length():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)

    def truncated(chunk_id):
        return bytearray(b"ab")

    with pytest.raises(bundle.BundleError):
        cloudpickle.load_bundle(main, manifest, truncated)


def test_out_of_order_manifest():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    manifest.entries[0].index = 9
    with pytest.raises(bundle.ChunkOrderError):
        cloudpickle.load_bundle(main, manifest, storage.writable)


def test_missing_entry_gap_detected():
    storage = Storage()
    first = PickleBuffer(bytearray(b"abcdef"))
    second = PickleBuffer(bytearray(b"ghijkl"))
    main, manifest = dump_to_storage([first, second], storage)
    # Drop the first entry: the remaining index is then wrong / a buffer gap
    # exists between stream and manifest.
    manifest.entries = [manifest.entries[1]]
    with pytest.raises(bundle.ChunkOrderError):
        cloudpickle.load_bundle(main, manifest, storage.writable)


def test_buffer_count_mismatch():
    # Stream expects one buffer but an empty manifest is supplied.
    storage = Storage()
    main, _ = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    empty = bundle.BufferManifest([])
    with pytest.raises(bundle.ChunkOrderError):
        cloudpickle.load_bundle(main, empty, storage.writable)


def test_total_size_limit_declared():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    with pytest.raises(bundle.BufferLimitError):
        cloudpickle.load_bundle(
            main, manifest, storage.writable, max_total_size=2
        )


def test_readonly_source_for_writable_chunk():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    # Zero copy cannot satisfy a writable buffer from a read-only source.
    with pytest.raises(bundle.BufferReadOnlyError):
        cloudpickle.load_bundle(main, manifest, storage.readonly)


def test_copy_materializes_writable_from_readonly():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    with cloudpickle.load_bundle(
        main, manifest, storage.readonly, copy=True
    ) as loaded:
        view = loaded.obj
        assert view.readonly is False
        view[0] = ord("Z")
        assert bytes(view) == b"Zbcdef"


def test_invalid_manifest_json():
    storage = Storage()
    main, _ = dump_to_storage(PickleBuffer(bytearray(b"abc")), storage)
    with pytest.raises(bundle.ManifestError):
        cloudpickle.load_bundle(main, b"{not json", storage.writable)


def test_inconsistent_alias_entries():
    storage = Storage()
    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    # Add a second entry claiming the same chunk id but disagreeing on length.
    first = manifest.entries[0]
    manifest.entries.append(
        bundle.BufferEntry(
            index=1,
            chunk_id=first.chunk_id,
            length=first.length + 1,
            digest=first.digest,
            readonly=first.readonly,
        )
    )
    with pytest.raises(bundle.ManifestError):
        cloudpickle.load_bundle(main, manifest, storage.writable)


# --------------------------------------------------------------------------- #
# Resource lifetime
# --------------------------------------------------------------------------- #


def test_failed_fetch_releases_borrowed_views():
    storage = Storage()
    original = [bytearray(b"abcdef"), bytearray(b"ghijkl")]
    main, manifest = dump_to_storage(
        [PickleBuffer(original[0]), PickleBuffer(original[1])], storage
    )
    borrowed = []

    def fail_on_second(chunk_id):
        # First chunk fetches successfully, then the second one fails.
        if not borrowed:
            view = memoryview(bytearray(storage.blobs[chunk_id]))
            borrowed.append(view)
            return view
        raise RuntimeError("boom")

    with pytest.raises(bundle.MissingChunkError):
        cloudpickle.load_bundle(main, manifest, fail_on_second)
    # The already-borrowed view must have been released.
    assert all(view.release() is None for view in borrowed)
    with pytest.raises(ValueError):
        bytes(borrowed[0])


def test_zero_copy_holds_source_lifetime():
    storage = Storage()
    source = bytearray(b"abcdef")
    main, manifest = dump_to_storage(PickleBuffer(source), storage)

    def get_view(chunk_id):
        return memoryview(source)

    with cloudpickle.load_bundle(main, manifest, get_view) as loaded:
        view = loaded.obj
        assert view.obj is source
        view[0] = ord("Z")
        assert source[0] == ord("Z")
    # The zero-copy view is borrowed directly from the caller-owned memoryview;
    # releasing the holder must not release the caller's own view.
    assert bytes(view) == b"Zbcdef"


def test_zero_copy_releases_loader_created_readonly_views():
    # When the loader itself creates a read-only alias from a writable source,
    # that loader-created view is tracked and released on exit.
    storage = Storage()

    def writable_source(chunk_id):
        return memoryview(bytearray(storage.blobs[chunk_id]))

    main, manifest = dump_to_storage(PickleBuffer(b"readonly-data"), storage)
    with cloudpickle.load_bundle(main, manifest, writable_source) as loaded:
        view = loaded.obj
        assert view.readonly is True
        assert bytes(view) == b"readonly-data"
    # The loader-created read-only alias is released; the backing storage (a
    # fresh bytearray created in the getter) remains untouched and owned by the
    # getter side.
    with pytest.raises(ValueError):
        bytes(view)


def test_copy_mode_does_not_keep_source_alive():
    storage = Storage()
    holder = {}

    class WeakRefableBuffer:
        __slots__ = ("_data", "__weakref__")

        def __init__(self, data):
            self._data = data

        def __buffer__(self, flags):
            return memoryview(self._data)

    def make_source(chunk_id):
        source = WeakRefableBuffer(bytearray(storage.blobs[chunk_id]))
        holder["ref"] = weakref.ref(source)
        return source

    main, manifest = dump_to_storage(PickleBuffer(bytearray(b"abcdef")), storage)
    with cloudpickle.load_bundle(
        main, manifest, make_source, copy=True
    ) as loaded:
        view = loaded.obj
        assert bytes(view) == b"abcdef"
    gc.collect()
    assert holder["ref"]() is None  # source can be reclaimed after load


def test_bundle_close_invalidates_views():
    data = bytearray(b"lifecycle")
    b = make_bundle([PickleBuffer(data)])
    views = [view for _, view in b.chunks()]
    assert bytes(views[0]) == b"lifecycle"
    b.close()
    with pytest.raises(ValueError):
        bytes(views[0])
    # Double close is safe.
    b.close()


def test_iterating_closed_bundle_fails():
    b = make_bundle(PickleBuffer(bytearray(b"x")))
    b.close()
    with pytest.raises(bundle.BundleError):
        list(b.chunks())


# --------------------------------------------------------------------------- #
# API surface / compatibility
# --------------------------------------------------------------------------- #


def test_protocol_below_five_rejected():
    with pytest.raises(ValueError):
        cloudpickle.dumps_bundle([1], protocol=4)


def test_manifest_json_roundtrip():
    storage = Storage()
    _, manifest = dump_to_storage(
        [PickleBuffer(bytearray(b"abc")), PickleBuffer(b"def")], storage
    )
    restored = bundle.BufferManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.total_length == 6


def test_regular_dumps_loads_unchanged():
    restored = cloudpickle.loads(cloudpickle.dumps(lambda x: x + 1))
    assert restored(41) == 42


def test_public_api_exported():
    for name in [
        "dump_bundle",
        "dumps_bundle",
        "load_bundle",
        "BufferEntry",
        "BufferManifest",
        "BufferBundle",
        "LoadedObject",
        "BundleError",
        "MissingChunkError",
        "ChunkOrderError",
        "BufferTamperedError",
        "BufferLimitError",
        "BufferReadOnlyError",
    ]:
        assert hasattr(cloudpickle, name)
