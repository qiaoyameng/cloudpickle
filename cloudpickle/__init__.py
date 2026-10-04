from . import cloudpickle
from .cloudpickle import *  # noqa

__doc__ = cloudpickle.__doc__

__version__ = "3.2.0.dev0"

__all__ = [  # noqa
    "__version__",
    "Pickler",
    "CloudPickler",
    "ClassIdentityNamespace",
    "dumps",
    "loads",
    "dump",
    "load",
    "get_class_identity_namespace",
    "register_pickle_by_value",
    "unregister_pickle_by_value",
]
