# cloudpickle

[![Automated Tests](https://github.com/cloudpipe/cloudpickle/workflows/Automated%20Tests/badge.svg?branch=master&event=push)](https://github.com/cloudpipe/cloudpickle/actions)
[![codecov.io](https://codecov.io/github/cloudpipe/cloudpickle/coverage.svg?branch=master)](https://codecov.io/github/cloudpipe/cloudpickle?branch=master)

`cloudpickle` makes it possible to serialize Python constructs not supported
by the default `pickle` module from the Python standard library.

`cloudpickle` is especially useful for **cluster computing** where Python
code is shipped over the network to execute on remote hosts, possibly close
to the data.

Among other things, `cloudpickle` supports pickling for **lambda functions**
along with **functions and classes defined interactively** in the
`__main__` module (for instance in a script, a shell or a Jupyter notebook).

Cloudpickle can only be used to send objects between the **exact same version
of Python**.

Using `cloudpickle` for **long-term object storage is not supported and
strongly discouraged.**

**Security notice**: one should **only load pickle data from trusted sources** as
otherwise `pickle.load` can lead to arbitrary code execution resulting in a critical
security vulnerability.


Installation
------------

The latest release of `cloudpickle` is available from
[pypi](https://pypi.python.org/pypi/cloudpickle):

    pip install cloudpickle


Examples
--------

Pickling a lambda expression:

```python
>>> import cloudpickle
>>> squared = lambda x: x ** 2
>>> pickled_lambda = cloudpickle.dumps(squared)

>>> import pickle
>>> new_squared = pickle.loads(pickled_lambda)
>>> new_squared(2)
4
```

Pickling a function interactively defined in a Python shell session
(in the `__main__` module):

```python
>>> CONSTANT = 42
>>> def my_function(data: int) -> int:
...     return data + CONSTANT
...
>>> pickled_function = cloudpickle.dumps(my_function)
>>> depickled_function = pickle.loads(pickled_function)
>>> depickled_function
<function __main__.my_function(data:int) -> int>
>>> depickled_function(43)
85
```


Overriding pickle's serialization mechanism for importable constructs:
----------------------------------------------------------------------

An important difference between `cloudpickle` and `pickle` is that
`cloudpickle` can serialize a function or class **by value**, whereas `pickle`
can only serialize it **by reference**. Serialization by reference treats
functions and classes as attributes of modules, and pickles them through
instructions that trigger the import of their module at load time.
Serialization by reference is thus limited in that it assumes that the module
containing the function or class is available/importable in the unpickling
environment. This assumption breaks when pickling constructs defined in an
interactive session, a case that is automatically detected by `cloudpickle`,
that pickles such constructs **by value**.

Another case where the importability assumption is expected to break is when
developing a module in a distributed execution environment: the worker
processes may not have access to the said module, for example if they live on a
different machine than the process in which the module is being developed. By
itself, `cloudpickle` cannot detect such "locally importable" modules and
switch to serialization by value; instead, it relies on its default mode, which
is serialization by reference. However, since `cloudpickle 2.0.0`, one can
explicitly specify modules for which serialization by value should be used,
using the
`register_pickle_by_value(module)`/`/unregister_pickle_by_value(module)` API:

```python
>>> import cloudpickle
>>> import my_module
>>> cloudpickle.register_pickle_by_value(my_module)
>>> cloudpickle.dumps(my_module.my_function)  # my_function is pickled by value
>>> cloudpickle.unregister_pickle_by_value(my_module)
>>> cloudpickle.dumps(my_module.my_function)  # my_function is pickled by reference
```

Using this API, there is no need to re-install the new version of the module on
all the worker nodes nor to restart the workers: restarting the client Python
process with the new source code is enough.

Note that this feature is still **experimental**, and may fail in the following
situations:

- If the body of a function/class pickled by value contains an `import` statement:
  ```python
  >>> def f():
  >>> ... from another_module import g
  >>> ... # calling f in the unpickling environment may fail if another_module
  >>> ... # is unavailable
  >>> ... return g() + 1
  ```

- If a function pickled by reference uses a function pickled by value during its execution.


Out-of-band buffer bundles (protocol 5)
---------------------------------------

When the object graph carries large protocol 5 buffers (objects that expose a
`pickle.PickleBuffer`, such as NumPy arrays), it is often desirable to keep
those large payloads out of the main pickle stream and store them separately.
`cloudpickle` provides a small "bundle" API for this. Only the Python standard
library (`PickleBuffer` and `memoryview`) is needed; NumPy is not required.

```python
>>> import cloudpickle
>>> from pickle import PickleBuffer

>>> payload = bytearray(b"x" * 1_000_000)  # any buffer-protocol object
>>> obj = {"data": PickleBuffer(payload), "meta": 42}

>>> storage = {}  # anything addressable by id (object store, files, ...)
>>> with cloudpickle.dump_bundle(obj) as bundle:
...     main = bundle.main              # small pickle stream, no buffer data
...     manifest = bundle.manifest      # order, length, digest, read-only
...     for chunk_id, view in bundle:   # iterable buffer contents
...         storage[chunk_id] = bytes(view)
```

On the receiving side, the caller provides a `get_chunk(chunk_id)` function.
The loader checks the manifest for missing chunks, ordering, digest (content
tampering) and a total-size limit, and honors the read-only constraint, all
*before* running the trusted pickle restore:

```python
>>> def get_chunk(chunk_id):
...     return storage[chunk_id]

>>> with cloudpickle.load_bundle(main, manifest, get_chunk) as loaded:
...     loaded.obj["meta"]
...     bytes(loaded.obj["data"][:5])
42
b'xxxxx'
```

The returned context manager owns the zero-copy buffer lifetimes; borrowed
buffers are released when the block exits. Pass `copy=True` to materialize
independent writable copies instead. The regular `cloudpickle.dumps` and
`cloudpickle.loads` functions are unchanged and still inline the buffers.

Alias handling: when the *same underlying buffer* is referenced from several
places, the manifest shares one chunk id and the restored object keeps views of
the same memory. Two originally independent, writable buffers that merely
contain equal bytes are never de-duplicated, so writing through one restored
view can never mutate the other.


Running the tests
-----------------

- With `tox`, to test run the tests for all the supported versions of
  Python and PyPy:

      pip install tox
      tox

  or alternatively for a specific environment:

      tox -e py312


- With `pytest` to only run the tests for your current version of
  Python:

      pip install -r dev-requirements.txt
      PYTHONPATH='.:tests' pytest

History
-------

`cloudpickle` was initially developed by [picloud.com](http://web.archive.org/web/20140721022102/http://blog.picloud.com/2013/11/17/picloud-has-joined-dropbox/) and shipped as part of
the client SDK.

A copy of `cloudpickle.py` was included as part of PySpark, the Python
interface to [Apache Spark](https://spark.apache.org/). Davies Liu, Josh
Rosen, Thom Neale and other Apache Spark developers improved it significantly,
most notably to add support for PyPy and Python 3.

The aim of the `cloudpickle` project is to make that work available to a wider
audience outside of the Spark ecosystem and to make it easier to improve it
further notably with the help of a dedicated non-regression test suite.
