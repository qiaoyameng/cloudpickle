"""Tests for the optional source bundle feature of cloudpickle.

A source bundle is a caller-provided mapping of filenames to their source
text. When pickling dynamic functions or classes, the sources of the files
that define them are attached to the payload and registered in linecache at
unpickling time so that tracebacks and inspect can display source lines on
processes where those files are not available.
"""

import importlib.util
import inspect
import linecache
import os
import pickle
import subprocess
import sys

import pytest

import cloudpickle


def _write_file(directory, name, content):
    path = os.path.join(str(directory), name)
    with open(path, "w") as f:
        f.write(content)
    return path


def _import_from_path(path, module_name):
    """Import a module from a path without registering it in sys.modules.

    Functions and classes of such a module are pickled by value, as dynamic
    objects, which is what the source bundle feature targets.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_child(script, *args):
    """Run a child Python process simulating the remote (receiver) side."""
    repo_folder = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_folder + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


_CHILD_CALL_AND_PRINT_TRACEBACK = """
import pickle, sys, traceback
with open(sys.argv[1], "rb") as f:
    func = pickle.load(f)
try:
    func()
except Exception:
    traceback.print_exc()
"""

_CHILD_CALL_LIST_AND_PRINT_TRACEBACKS = """
import pickle, sys, traceback
with open(sys.argv[1], "rb") as f:
    funcs = pickle.load(f)
for func in funcs:
    try:
        func()
    except Exception:
        traceback.print_exc()
"""

_CHILD_CALL_METHOD_AND_PRINT_TRACEBACK = """
import pickle, sys, traceback
with open(sys.argv[1], "rb") as f:
    cls = pickle.load(f)
try:
    cls().greet()
except Exception:
    traceback.print_exc()
"""


@pytest.fixture
def linecache_cleanup():
    """Restore linecache entries modified by a test."""
    snapshot = dict(linecache.cache)
    try:
        yield
    finally:
        linecache.cache.clear()
        linecache.cache.update(snapshot)


class TestSourceBundle:
    def test_traceback_shows_bundled_source(self, tmp_path):
        source = 'def boom():\n    raise ValueError("kaboom")\n'
        path = _write_file(tmp_path, "mymod.py", source)
        mod = _import_from_path(path, "mymod_basic")
        payload = cloudpickle.dumps(mod.boom, source_bundle={path: source})
        # Simulate the receiver side: the source file is not available.
        os.remove(path)
        pickle_file = _write_file(tmp_path, "func.pkl", "")
        with open(pickle_file, "wb") as f:
            f.write(payload)
        proc = _run_child(_CHILD_CALL_AND_PRINT_TRACEBACK, pickle_file)
        assert proc.returncode == 0, proc.stderr
        tb = proc.stderr
        assert f'File "{path}", line 2, in boom' in tb
        assert 'raise ValueError("kaboom")' in tb
        assert "ValueError: kaboom" in tb

    def test_two_same_named_files(self, tmp_path):
        # Two files with the same basename in different packages, each
        # defining a function: each traceback must display the source of
        # its own file.
        dir_a = tmp_path / "pkg_a"
        dir_b = tmp_path / "pkg_b"
        dir_a.mkdir()
        dir_b.mkdir()
        source_a = "def func():\n    raise ValueError('from a')\n"
        source_b = "\n\n\ndef func():\n    raise ValueError('from b')\n"
        path_a = _write_file(dir_a, "utils.py", source_a)
        path_b = _write_file(dir_b, "utils.py", source_b)
        mod_a = _import_from_path(path_a, "pkg_a_utils")
        mod_b = _import_from_path(path_b, "pkg_b_utils")
        payload = cloudpickle.dumps(
            [mod_a.func, mod_b.func],
            source_bundle={path_a: source_a, path_b: source_b},
        )
        os.remove(path_a)
        os.remove(path_b)
        pickle_file = _write_file(tmp_path, "funcs.pkl", "")
        with open(pickle_file, "wb") as f:
            f.write(payload)
        proc = _run_child(_CHILD_CALL_LIST_AND_PRINT_TRACEBACKS, pickle_file)
        assert proc.returncode == 0, proc.stderr
        tb = proc.stderr
        # func from pkg_a raises at line 2, func from pkg_b at line 5.
        assert f'File "{path_a}", line 2, in func' in tb
        assert "raise ValueError('from a')" in tb
        assert f'File "{path_b}", line 5, in func' in tb
        assert "raise ValueError('from b')" in tb
        # The two sources are not mixed up.
        block_a, block_b = tb.split("ValueError: from a")
        assert "from b" not in block_a
        assert "from a" not in block_b

    def test_nested_function_traceback(self, tmp_path):
        source = (
            "def outer():\n"
            "    def inner():\n"
            '        raise RuntimeError("nested boom")  # marker-7f3d2\n'
            "    return inner()\n"
        )
        path = _write_file(tmp_path, "nested_mod.py", source)
        mod = _import_from_path(path, "nested_mod")
        payload = cloudpickle.dumps(mod.outer, source_bundle={path: source})
        # The nested code object references the same file as its parent:
        # the source is attached only once in the payload.
        assert payload.count(b"marker-7f3d2") == 1
        os.remove(path)
        pickle_file = _write_file(tmp_path, "func.pkl", "")
        with open(pickle_file, "wb") as f:
            f.write(payload)
        proc = _run_child(_CHILD_CALL_AND_PRINT_TRACEBACK, pickle_file)
        assert proc.returncode == 0, proc.stderr
        tb = proc.stderr
        assert f'File "{path}", line 4, in outer' in tb
        assert "return inner()" in tb
        assert f'File "{path}", line 3, in inner' in tb
        assert 'raise RuntimeError("nested boom")' in tb

    def test_class_method_traceback(self, tmp_path):
        source = (
            "class Greeter:\n"
            "    def greet(self):\n"
            "        raise ValueError('class boom')\n"
        )
        path = _write_file(tmp_path, "class_mod.py", source)
        mod = _import_from_path(path, "class_mod")
        payload = cloudpickle.dumps(mod.Greeter, source_bundle={path: source})
        os.remove(path)
        pickle_file = _write_file(tmp_path, "cls.pkl", "")
        with open(pickle_file, "wb") as f:
            f.write(payload)
        proc = _run_child(_CHILD_CALL_METHOD_AND_PRINT_TRACEBACK, pickle_file)
        assert proc.returncode == 0, proc.stderr
        tb = proc.stderr
        assert f'File "{path}", line 3, in greet' in tb
        assert "raise ValueError('class boom')" in tb

    def test_missing_source_degrades_gracefully(self, tmp_path):
        source = 'def boom():\n    raise ValueError("no source")\n'
        path = _write_file(tmp_path, "plain_mod.py", source)
        mod = _import_from_path(path, "plain_mod")
        # No source bundle at all, and a bundle that does not cover the
        # file: both must keep working with the historical behavior.
        payloads = [
            cloudpickle.dumps(mod.boom),
            cloudpickle.dumps(mod.boom, source_bundle={"/other.py": "x = 1\n"}),
        ]
        os.remove(path)
        for i, payload in enumerate(payloads):
            pickle_file = _write_file(tmp_path, f"func{i}.pkl", "")
            with open(pickle_file, "wb") as f:
                f.write(payload)
            proc = _run_child(_CHILD_CALL_AND_PRINT_TRACEBACK, pickle_file)
            assert proc.returncode == 0, proc.stderr
            tb = proc.stderr
            assert 'File "<dynamic-code>", line 2, in boom' in tb
            assert 'raise ValueError("no source")' not in tb
            assert "ValueError: no source" in tb

    def test_source_is_display_only(self, tmp_path, linecache_cleanup):
        # The restored function still executes the pickled code object: the
        # bundled source is only used for display, never re-executed.
        real_source = "def answer():\n    return 42\n"
        doctored_source = "def answer():\n    return 0  # doctored\n"
        path = _write_file(tmp_path, "display_mod.py", real_source)
        mod = _import_from_path(path, "display_mod")
        payload = cloudpickle.dumps(mod.answer, source_bundle={path: doctored_source})
        os.remove(path)
        func = pickle.loads(payload)
        assert func() == 42
        assert func.__code__.co_filename == path
        assert linecache.getline(path, 2) == "    return 0  # doctored\n"

    def test_inspect_getsource(self, tmp_path, linecache_cleanup):
        source = "def greet(name):\n    return 'hello ' + name\n"
        path = _write_file(tmp_path, "inspect_mod.py", source)
        mod = _import_from_path(path, "inspect_mod")
        payload = cloudpickle.dumps(mod.greet, source_bundle={path: source})
        os.remove(path)
        func = pickle.loads(payload)
        assert inspect.getsource(func) == source

    def test_virtual_path_when_linecache_entry_conflicts(
        self, tmp_path, linecache_cleanup
    ):
        source = "def answer():\n    return 42\n"
        path = _write_file(tmp_path, "conflict_mod.py", source)
        mod = _import_from_path(path, "conflict_mod")
        payload = cloudpickle.dumps(mod.answer, source_bundle={path: source})
        os.remove(path)
        # Simulate a receiver where linecache already holds different
        # content for this filename (e.g. a same-named file from another
        # package).
        other_source = "def answer():\n    return 0\n"
        other_lines = other_source.splitlines(keepends=True)
        linecache.cache[path] = (len(other_source), None, other_lines, path)
        func = pickle.loads(payload)
        virtual_filename = func.__code__.co_filename
        # An independent virtual path is allocated...
        assert virtual_filename != path
        assert virtual_filename.startswith("<cloudpickle-source-bundle:")
        assert virtual_filename.endswith("conflict_mod.py")
        # ...the pre-existing linecache entry is not overwritten...
        assert linecache.cache[path][2] == other_lines
        # ...and the bundled source is available under the virtual path.
        assert linecache.getlines(virtual_filename) == source.splitlines(
            keepends=True
        )
        # Unpickling again reuses the same virtual path (idempotent).
        func2 = pickle.loads(payload)
        assert func2.__code__.co_filename == virtual_filename

    def test_virtual_path_when_file_exists_on_receiver(
        self, tmp_path, linecache_cleanup
    ):
        source = "def answer():\n    return 42\n"
        path = _write_file(tmp_path, "ondisk_mod.py", source)
        mod = _import_from_path(path, "ondisk_mod")
        payload = cloudpickle.dumps(mod.answer, source_bundle={path: source})
        # The receiver has a different file at the same location.
        other_source = "def answer():\n    return 0\n"
        with open(path, "w") as f:
            f.write(other_source)
        linecache.cache.pop(path, None)
        func = pickle.loads(payload)
        virtual_filename = func.__code__.co_filename
        assert virtual_filename != path
        assert virtual_filename.startswith("<cloudpickle-source-bundle:")
        # The on-disk file is still the one served for the real filename.
        assert "".join(linecache.getlines(path)) == other_source
        assert "".join(linecache.getlines(virtual_filename)) == source

    def test_same_content_on_receiver_reuses_filename(
        self, tmp_path, linecache_cleanup
    ):
        source = "def answer():\n    return 42\n"
        path = _write_file(tmp_path, "same_mod.py", source)
        mod = _import_from_path(path, "same_mod")
        payload = cloudpickle.dumps(mod.answer, source_bundle={path: source})
        # The receiver has the exact same file: no virtual path is needed.
        linecache.cache.pop(path, None)
        func = pickle.loads(payload)
        assert func.__code__.co_filename == path
        assert func() == 42

    def test_bytes_source_with_encoding_cookie(self, tmp_path, linecache_cleanup):
        source = "# -*- coding: latin-1 -*-\ndef answer():\n    return 'caf\xe9'\n"
        path = os.path.join(str(tmp_path), "latin_mod.py")
        with open(path, "wb") as f:
            f.write(source.encode("latin-1"))
        mod = _import_from_path(path, "latin_mod")
        payload = cloudpickle.dumps(
            mod.answer, source_bundle={path: source.encode("latin-1")}
        )
        os.remove(path)
        func = pickle.loads(payload)
        assert func() == "caf\xe9"
        assert linecache.getline(path, 3) == "    return 'caf\xe9'\n"

    def test_invalid_source_type_raises(self, tmp_path):
        path = _write_file(tmp_path, "typed_mod.py", "def f():\n    pass\n")
        mod = _import_from_path(path, "typed_mod")
        with pytest.raises(TypeError, match="str or bytes"):
            cloudpickle.dumps(mod.f, source_bundle={path: 42})
        with pytest.raises(TypeError, match="filenames should be strings"):
            cloudpickle.dumps(mod.f, source_bundle={42: "x = 1\n"})

    def test_undecodable_source_raises(self, tmp_path):
        source = "def f():\n    pass\n"
        path = _write_file(tmp_path, "undecodable_mod.py", source)
        mod = _import_from_path(path, "undecodable_mod")
        with pytest.raises(ValueError, match="cannot decode"):
            cloudpickle.dumps(mod.f, source_bundle={path: b"\xff\xfe\xfd"})

    def test_line_range_check(self, tmp_path):
        source = "\n\n\n\ndef f():\n    pass\n"
        path = _write_file(tmp_path, "range_mod.py", source)
        mod = _import_from_path(path, "range_mod")
        # The bundled source is too short to cover the code object lines.
        with pytest.raises(ValueError, match="spans lines"):
            cloudpickle.dumps(mod.f, source_bundle={path: "def f():\n    pass\n"})

    def test_source_bundle_budget(self, tmp_path):
        source = "def f():\n    return 42\n"
        path = _write_file(tmp_path, "budget_mod.py", source)
        mod = _import_from_path(path, "budget_mod")
        with pytest.raises(ValueError, match="total budget"):
            cloudpickle.dumps(
                mod.f, source_bundle={path: source}, source_bundle_max_bytes=10
            )
        with pytest.raises(ValueError, match="positive integer"):
            cloudpickle.dumps(
                mod.f, source_bundle={path: source}, source_bundle_max_bytes=0
            )
        # A large enough budget accepts the same bundle.
        cloudpickle.dumps(
            mod.f, source_bundle={path: source}, source_bundle_max_bytes=10**6
        )
