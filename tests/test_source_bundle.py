import io
import inspect
import linecache
import pickle
import pickletools
import textwrap
import traceback
import types

import pytest

import cloudpickle


def _compile_source(filename, source):
    namespace = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _format_exception(function):
    try:
        function()
    except RuntimeError:
        return traceback.format_exc()
    raise AssertionError("function did not raise")


def test_same_named_files_get_independent_virtual_sources():
    first_source = textwrap.dedent(
        """\
        def f():
            raise RuntimeError("first")
        """
    )
    second_source = textwrap.dedent(
        """\
        def f():
            raise RuntimeError("second")
        """
    )
    first_filename = "/pkg_one/module.py"
    second_filename = "/pkg_two/module.py"
    first_function = _compile_source(first_filename, first_source)["f"]
    second_function = _compile_source(second_filename, second_source)["f"]

    restored = pickle.loads(
        cloudpickle.dumps(
            (first_function, second_function),
            source_bundle={
                first_filename: first_source,
                second_filename: second_source,
            },
        )
    )

    assert restored[0].__code__.co_filename != restored[1].__code__.co_filename
    assert restored[0].__code__.co_filename.startswith("<cloudpickle-source:")
    assert restored[1].__code__.co_filename.startswith("<cloudpickle-source:")
    assert linecache.getline(restored[0].__code__.co_filename, 2) == (
        '    raise RuntimeError("first")\n'
    )
    assert linecache.getline(restored[1].__code__.co_filename, 2) == (
        '    raise RuntimeError("second")\n'
    )
    assert 'raise RuntimeError("second")' in _format_exception(restored[1])


def test_nested_code_objects_share_remapped_source():
    source = textwrap.dedent(
        """\
        def outer():
            def inner():
                raise RuntimeError("nested")
            return inner()
        """
    )
    filename = "/nested/module.py"
    function = _compile_source(filename, source)["outer"]

    restored = pickle.loads(
        cloudpickle.dumps(function, source_bundle={filename: source})
    )
    nested_code = next(
        const
        for const in restored.__code__.co_consts
        if isinstance(const, types.CodeType)
    )

    assert nested_code.co_filename == restored.__code__.co_filename
    assert restored.__code__.co_code == function.__code__.co_code
    assert nested_code.co_code == next(
        const
        for const in function.__code__.co_consts
        if isinstance(const, types.CodeType)
    ).co_code
    assert inspect.getsource(restored).splitlines()[0] == "def outer():"
    formatted = _format_exception(restored)
    assert 'line 3, in inner' in formatted
    assert 'raise RuntimeError("nested")' in formatted
    assert 'line 4, in outer' in formatted
    assert "    return inner()" in formatted


def test_source_text_is_never_executed_to_rebuild_function():
    executable_source = "def f():\n    return 1\n"
    display_source = (
        'raise RuntimeError("source was executed")\n'
        "def f():\n"
        "    return 2\n"
    )
    function = _compile_source("/no-exec/module.py", executable_source)["f"]

    restored = pickle.loads(
        cloudpickle.dumps(
            function,
            source_bundle={"/no-exec/module.py": display_source},
        )
    )

    assert restored() == 1
    assert restored.__code__.co_code == function.__code__.co_code


def test_missing_source_falls_back_without_overwriting_linecache():
    source = textwrap.dedent(
        """\
        def f():
            raise RuntimeError("missing")
        """
    )
    filename = "/missing/module.py"
    function = _compile_source(filename, source)["f"]

    restored = pickle.loads(cloudpickle.dumps(function, source_bundle={}))

    assert restored.__code__.co_filename == "<dynamic-code>"
    assert linecache.getline("<dynamic-code>", 2) == ""
    formatted = _format_exception(restored)
    assert 'File "<dynamic-code>", line 2, in f' in formatted
    assert 'raise RuntimeError("missing")' not in formatted


def test_existing_linecache_entry_is_not_overwritten():
    source = textwrap.dedent(
        """\
        def f():
            raise RuntimeError("bundled")
        """
    )
    filename = "/existing/module.py"
    function = _compile_source(filename, source)["f"]
    existing_lines = ["EXISTING LINE\n"]
    linecache.cache["<cloudpickle-source-existing>"] = (
        1,
        None,
        existing_lines,
        "<cloudpickle-source-existing>",
    )

    restored = pickle.loads(
        cloudpickle.dumps(function, source_bundle={filename: source})
    )

    assert restored.__code__.co_filename != "<cloudpickle-source-existing>"
    assert linecache.cache["<cloudpickle-source-existing>"][2] is existing_lines
    linecache.cache.pop("<cloudpickle-source-existing>", None)


def test_dynamic_class_methods_use_bundled_source():
    source = textwrap.dedent(
        """\
        class Example:
            def method(self):
                raise RuntimeError("method")
        """
    )
    filename = "/class/module.py"
    example_class = _compile_source(filename, source)["Example"]

    restored_class = pickle.loads(
        cloudpickle.dumps(example_class, source_bundle={filename: source})
    )

    assert inspect.getsource(restored_class.method) == "".join(
        source.splitlines(True)[1:]
    )
    formatted = _format_exception(restored_class().method)
    assert 'raise RuntimeError("method")' in formatted


def test_source_bundle_rejects_invalid_encoding():
    function = _compile_source(
        "/invalid/module.py", "def f():\n    return 1\n"
    )["f"]
    with pytest.raises(pickle.PicklingError, match="valid UTF-8"):
        cloudpickle.dumps(
            function, source_bundle={"/invalid/module.py": b"\xff"}
        )


def test_source_bundle_checks_line_number_range():
    source = "def f():\n    return 1\n"
    function = _compile_source("/short/module.py", source)["f"]
    with pytest.raises(pickle.PicklingError, match="does not contain line"):
        cloudpickle.dumps(
            function, source_bundle={"/short/module.py": "def f():\n"}
        )


def test_source_bundle_enforces_total_source_budget():
    function = _compile_source(
        "/budget/module.py", "def f():\n    return 1\n"
    )["f"]
    with pytest.raises(pickle.PicklingError, match="maximum source size"):
        cloudpickle.dumps(
            function,
            source_bundle={"/budget/module.py": "x" * 10},
            max_source_bytes=9,
        )


def test_identical_source_text_is_counted_once():
    source = "x" * 10
    function = _compile_source(
        "/shared/module.py", "def f():\n    return 1\n"
    )["f"]
    payload = cloudpickle.dumps(
        function,
        source_bundle={"/shared/one.py": source, "/shared/two.py": source},
        max_source_bytes=10,
    )
    source_opcodes = [
        opcode
        for opcode, arg, _ in pickletools.genops(io.BytesIO(payload))
        if opcode.name == "SHORT_BINUNICODE" and arg == source
    ]
    assert len(source_opcodes) == 1
