"""Helpers for distinguishing missing optional packages from broken imports.

Optional model stacks import many transitive modules.  A broad ``ImportError``
handler therefore cannot safely conclude that the package named by Keep is
absent: the package may be installed while one of its dependencies is missing
or version-incompatible.  These helpers keep that distinction consistent
between provider construction and setup-time availability probes.
"""

import importlib
from typing import Literal


OptionalDependencyStatus = Literal["available", "missing", "broken"]


def is_missing_optional_dependency(
    error: BaseException,
    module_name: str,
) -> bool:
    """Return whether *error* means the requested top-level module is absent.

    ``ModuleNotFoundError.name`` identifies the module Python could not find.
    Only an exact match for the requested top-level package is considered
    missing.  A different name means an installed package failed while loading
    a transitive dependency and its original error must remain visible.
    """
    top_level = module_name.split(".", 1)[0]
    return isinstance(error, ModuleNotFoundError) and error.name == top_level


def probe_optional_dependency(
    module_name: str,
) -> tuple[OptionalDependencyStatus, str | None]:
    """Import *module_name* and classify availability without hiding failures."""
    try:
        importlib.import_module(module_name)
    except ImportError as error:
        if is_missing_optional_dependency(error, module_name):
            return "missing", None
        return "broken", str(error)
    return "available", None
