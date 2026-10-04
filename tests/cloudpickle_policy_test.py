import io
import pickle
import sys
import types

import pytest

import cloudpickle
from cloudpickle import ModulePicklePolicy
from cloudpickle.cloudpickle import Pickler
from cloudpickle.cloudpickle import register_pickle_by_value
from cloudpickle.cloudpickle import unregister_pickle_by_value


def _dumps(obj, **kwargs):
    file = io.BytesIO()
    Pickler(file, **kwargs).dump(obj)
    return file.getvalue()


def _module(name, source=""):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    if source:
        exec(source, module.__dict__)
    return module


def test_pickler_module_policy_covers_functions_classes_modules_and_typevars():
    module = _module(
        "cloudpickle_policy_objects",
        """
def function():
    return 1

class Class:
    pass

T = __import__("typing").TypeVar("T")
""",
    )
    try:
        by_reference = _dumps(module.function)
        assert b"<dynamic-code>" not in by_reference

        policy = {module.__name__: ModulePicklePolicy.BY_VALUE}
        assert b"<dynamic-code>" in _dumps(module.function, module_policy=policy)
        assert b"_make_skeleton_class" in _dumps(
            module.Class, module_policy=policy
        )
        assert b"_make_typevar" in _dumps(module.T, module_policy=policy)
        assert b"subimport" in _dumps(module)
        assert b"dynamic_subimport" in _dumps(module, module_policy=policy)
    finally:
        sys.modules.pop(module.__name__, None)


def test_pickler_module_policy_uses_specific_submodule_rule():
    package = _module("cloudpickle_policy_specific_package")
    child = _module(
        "cloudpickle_policy_specific_package.child",
        "def function(): return 1\n",
    )
    package.child = child
    try:
        pickled = _dumps(
            child.function,
            module_policy={
                package.__name__: ModulePicklePolicy.BY_VALUE,
                child.__name__: ModulePicklePolicy.BY_REFERENCE,
            },
        )
        assert b"<dynamic-code>" not in pickled

        pickled = _dumps(
            child.function,
            module_policy={
                package.__name__: ModulePicklePolicy.BY_REFERENCE,
                child.__name__: ModulePicklePolicy.BY_VALUE,
            },
        )
        assert b"<dynamic-code>" in pickled
    finally:
        sys.modules.pop(child.__name__, None)
        sys.modules.pop(package.__name__, None)


def test_pickler_module_policy_default_overrides_global_registration():
    module = _module(
        "cloudpickle_policy_default",
        "def function(): return 1\n",
    )
    register_pickle_by_value(module)
    try:
        pickled = _dumps(
            module.function,
            module_policy={module.__name__: ModulePicklePolicy.DEFAULT},
        )
        assert b"<dynamic-code>" not in pickled
    finally:
        unregister_pickle_by_value(module)
        sys.modules.pop(module.__name__, None)


def test_pickler_module_policy_resolves_import_alias_by_identity():
    original = _module(
        "cloudpickle_policy_alias_original",
        "def function(): return 1\nclass Class: pass\n",
    )
    alias = _module("cloudpickle_policy_alias")
    alias.function = original.function
    alias.Class = original.Class
    try:
        pickled = _dumps(
            alias.function,
            module_policy={alias.__name__: ModulePicklePolicy.BY_VALUE},
        )
        assert b"<dynamic-code>" not in pickled

        pickled = _dumps(
            alias.function,
            module_policy={original.__name__: ModulePicklePolicy.BY_VALUE},
        )
        assert b"<dynamic-code>" in pickled
    finally:
        sys.modules.pop(alias.__name__, None)
        sys.modules.pop(original.__name__, None)


def test_pickler_module_policy_preserves_shared_globals_recursion_and_bases():
    module = _module(
        "cloudpickle_policy_shared_globals",
        """
def recursive(n):
    return n if n <= 1 else recursive(n - 1)

def calls_recursive():
    return recursive(3)

def shared():
    return 42
""",
    )
    base_module = _module(
        "cloudpickle_policy_base_module",
        "class Base: pass\n",
    )
    derived_module = _module(
        "cloudpickle_policy_derived_module",
        """
from cloudpickle_policy_base_module import Base
class Derived(Base):
    pass
""",
    )
    try:
        pickled = _dumps(
            [module.recursive, module.calls_recursive, module.shared],
            module_policy={module.__name__: ModulePicklePolicy.BY_VALUE},
        )
        recursive, caller, shared = pickle.loads(pickled)
        assert caller() == 1
        assert recursive(4) == 1
        assert caller.__globals__ is recursive.__globals__ is shared.__globals__

        pickled = _dumps(
            derived_module.Derived,
            module_policy={derived_module.__name__: ModulePicklePolicy.BY_VALUE},
        )
        Derived = pickle.loads(pickled)
        assert Derived.__mro__[1] is base_module.Base
    finally:
        sys.modules.pop(derived_module.__name__, None)
        sys.modules.pop(base_module.__name__, None)
        sys.modules.pop(module.__name__, None)


class _MutatePolicy:
    def __reduce__(self):
        register_pickle_by_value(sys.modules["cloudpickle_policy_frozen_global"])
        self.pickler.set_module_policy(
            "cloudpickle_policy_frozen_instance", ModulePicklePolicy.BY_VALUE
        )
        return dict, ((),)


def test_pickler_module_policy_is_fixed_when_dump_starts():
    global_module = _module(
        "cloudpickle_policy_frozen_global",
        "def function(): return 1\n",
    )
    instance_module = _module(
        "cloudpickle_policy_frozen_instance",
        "def function(): return 1\n",
    )
    file = io.BytesIO()
    pickler = Pickler(
        file,
        module_policy={global_module.__name__: ModulePicklePolicy.BY_REFERENCE},
    )
    mutator = _MutatePolicy()
    mutator.pickler = pickler
    try:
        pickler.dump([mutator, global_module.function, instance_module.function])
        assert file.getvalue().count(b"<dynamic-code>") == 0
    finally:
        registry = cloudpickle.list_registry_pickle_by_value()
        if global_module.__name__ in registry:
            unregister_pickle_by_value(global_module)
        sys.modules.pop(instance_module.__name__, None)
        sys.modules.pop(global_module.__name__, None)


def test_interleaved_pickler_module_policies_do_not_pollute_each_other():
    module_a = _module(
        "cloudpickle_policy_interleaved_a",
        "def function(): return 'a'\n",
    )
    module_b = _module(
        "cloudpickle_policy_interleaved_b",
        "def function(): return 'b'\n",
    )
    file_a = io.BytesIO()
    file_b = io.BytesIO()
    pickler_a = Pickler(
        file_a,
        module_policy={module_a.__name__: ModulePicklePolicy.BY_VALUE},
    )
    pickler_b = Pickler(
        file_b,
        module_policy={module_b.__name__: ModulePicklePolicy.BY_VALUE},
    )
    try:
        pickler_a.dump(module_a.function)
        pickler_b.dump(module_b.function)
        pickler_a.dump(module_b.function)
        pickler_b.dump(module_a.function)
        assert file_a.getvalue().count(b"<dynamic-code>") == 1
        assert file_b.getvalue().count(b"<dynamic-code>") == 1
    finally:
        sys.modules.pop(module_b.__name__, None)
        sys.modules.pop(module_a.__name__, None)
