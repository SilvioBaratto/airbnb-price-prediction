"""The dependency rule, enforced: inner layers never import outer ones, and no two layers
import each other.

Reads every module's imports with ``ast`` instead of trusting the README diagram, so a stray
import fails the build rather than slowly eroding the architecture.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "airbnb"

# Top-level modules and what each may import. `paths` is a leaf every layer that touches the
# disk may import: it imports nothing from the package, so it can never close a cycle.
TOP_LEVEL = {"__init__": set(), "__main__": {"cli"}, "paths": set()}

# Which airbnb layers each layer may import (itself always included).
ALLOWED = {
    "domain": set(),
    "application": {"domain"},
    "modeling": {"domain", "paths"},
    "datasource": {"domain", "paths"},
    "infrastructure": {"domain", "modeling", "paths"},
    "cli": {"domain", "application", "infrastructure", "datasource", "modeling", "paths"},
}


def _airbnb_imports(path: Path, package: Path = PACKAGE) -> set[str]:
    """Return the top-level airbnb names a module imports (``airbnb.x.y`` -> ``x``).

    Covers absolute imports, ``from airbnb import x`` and relative imports, which are resolved
    against the module's own position inside ``package``.
    """
    tree = ast.parse(path.read_text())
    here = list(path.relative_to(package.parent).with_suffix("").parts)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dotted = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - node.level]
                module = base + (node.module.split(".") if node.module else [])
            else:
                module = node.module.split(".") if node.module else []
            # `from airbnb import cli` names the layer in the alias, not in the module.
            dotted = (
                [module + [alias.name] for alias in node.names]
                if module == ["airbnb"]
                else [module]
            )
        else:
            continue
        for parts in dotted:
            if len(parts) > 1 and parts[0] == "airbnb":
                names.add(parts[1])
    return names


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_what_it_may(layer: str) -> None:
    """Each module in ``layer`` imports only itself and the layers it is allowed to see."""
    for module in sorted((PACKAGE / layer).rglob("*.py")):
        forbidden = _airbnb_imports(module) - ALLOWED[layer] - {layer}
        assert not forbidden, f"{module.relative_to(PACKAGE)} imports {sorted(forbidden)}"


def test_no_two_layers_may_import_each_other() -> None:
    """The allowed graph itself has no cycle, so passing the rule above means no cycle exists."""
    for layer, allowed in ALLOWED.items():
        for other in allowed & set(ALLOWED):
            assert layer not in ALLOWED[other], f"{layer} and {other} may import each other"


def test_top_level_modules_import_only_what_they_may() -> None:
    """``paths`` stays a leaf, ``__main__`` only reaches the CLI, and no top-level module hides."""
    modules = {p.stem for p in PACKAGE.glob("*.py")}
    assert modules == set(TOP_LEVEL)
    for name, allowed in TOP_LEVEL.items():
        forbidden = _airbnb_imports(PACKAGE / f"{name}.py") - allowed
        assert not forbidden, f"{name}.py imports {sorted(forbidden)}"


def test_every_layer_is_covered() -> None:
    """A new top-level package cannot dodge the rule by not being listed."""
    packages = {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    assert packages == set(ALLOWED)


@pytest.mark.parametrize(
    "source",
    [
        "from airbnb.cli import app",
        "import airbnb.cli.app",
        "from airbnb import cli",
        "from ..cli import app",
        "from .. import cli",
    ],
)
def test_every_import_style_is_seen(tmp_path: Path, source: str) -> None:
    """Absolute, package-alias and relative imports of a layer are all detected."""
    module = tmp_path / "airbnb" / "domain" / "rogue.py"
    module.parent.mkdir(parents=True)
    module.write_text(source + "\n")
    assert "cli" in _airbnb_imports(module, package=tmp_path / "airbnb")
