"""
conftest.py — pytest session setup.

PySpark and leidenalg are Databricks runtime dependencies not installed in the
local development / CI environment.  We stub them out so that all community
detector modules can be imported without a real Spark installation or Leiden
runtime.

Tests replace individual stubs with more specific MagicMocks via
unittest.mock tools.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


class _SparkColumn:
    """Minimal stub for a PySpark Column expression.

    PySpark Column objects return *Column expressions* (not Python bools) from
    comparison and arithmetic operators.  Python's ``MagicMock`` returns
    ``NotImplemented`` by default for those operators, which causes
    ``TypeError`` when two mock columns are compared (e.g.
    ``col("a") > col("b")``).

    This plain class avoids MagicMock entirely — every operator and every
    method call simply returns a new ``_SparkColumn`` — exactly what a real
    PySpark Column does.
    """

    def __getattr__(self, name):  # noqa: ANN001
        """Handle method chains: .alias(), .cast(), .contains(), .isin(), …"""
        return lambda *args, **kw: _SparkColumn()

    def __call__(self, *args, **kw):  # noqa: ANN001
        return _SparkColumn()

    # --- comparison ---
    def __gt__(self, other): return _SparkColumn()  # noqa: E704
    def __lt__(self, other): return _SparkColumn()  # noqa: E704
    def __ge__(self, other): return _SparkColumn()  # noqa: E704
    def __le__(self, other): return _SparkColumn()  # noqa: E704

    # --- boolean / bitwise ---
    def __and__(self, other): return _SparkColumn()   # noqa: E704
    def __rand__(self, other): return _SparkColumn()  # noqa: E704
    def __or__(self, other): return _SparkColumn()    # noqa: E704
    def __ror__(self, other): return _SparkColumn()   # noqa: E704
    def __invert__(self): return _SparkColumn()       # noqa: E704  (~col)

    # --- arithmetic ---
    def __add__(self, other): return _SparkColumn()      # noqa: E704
    def __radd__(self, other): return _SparkColumn()     # noqa: E704
    def __sub__(self, other): return _SparkColumn()      # noqa: E704
    def __rsub__(self, other): return _SparkColumn()     # noqa: E704
    def __mul__(self, other): return _SparkColumn()      # noqa: E704
    def __rmul__(self, other): return _SparkColumn()     # noqa: E704
    def __truediv__(self, other): return _SparkColumn()  # noqa: E704
    def __rtruediv__(self, other): return _SparkColumn() # noqa: E704
    def __neg__(self): return _SparkColumn()             # noqa: E704


class _SparkFunctions:
    """Minimal stub for the ``pyspark.sql.functions`` module.

    Every attribute access returns a callable that produces a ``_SparkColumn``
    instance, so expressions like ``col("a") > lit(1)`` work without error.
    """

    def __getattr__(self, name):  # noqa: ANN001
        return lambda *args, **kw: _SparkColumn()


def _stub_pyspark() -> None:
    """Insert minimal pyspark stubs into sys.modules if pyspark is absent."""
    try:
        import pyspark  # noqa: F401  # already installed — nothing to do
        return
    except ModuleNotFoundError:
        pass

    # Top-level package
    pyspark_stub = MagicMock()
    sys.modules.setdefault("pyspark", pyspark_stub)

    # pyspark.sql
    sql_stub = MagicMock()
    sys.modules.setdefault("pyspark.sql", sql_stub)

    # pyspark.sql.functions — every attribute is a callable returning _SparkColumn,
    # so column expressions such as col("a") > col("b") don't raise TypeError.
    sys.modules.setdefault("pyspark.sql.functions", _SparkFunctions())

    # pyspark.sql.types (imported transitively by some helpers)
    sys.modules.setdefault("pyspark.sql.types", MagicMock())


def _stub_leidenalg() -> None:
    """Insert a minimal leidenalg stub if the package is absent."""
    try:
        import leidenalg  # noqa: F401  # already installed — nothing to do
        return
    except ModuleNotFoundError:
        pass

    stub = MagicMock()
    sys.modules.setdefault("leidenalg", stub)


_stub_pyspark()
_stub_leidenalg()
