"""The dependency rule, enforced: inner layers never import outer ones.

Reads every module's imports with ``ast`` instead of trusting the README diagram, so a stray
import fails the build rather than slowly eroding the architecture.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "airbnb"

# Which airbnb layers each layer may import (itself always included).
ALLOWED = {
    "domain": set(),
    "application": {"domain"},
    "infrastructure": {"domain", "modeling"},
    "datasource": {"domain", "infrastructure"},
    "modeling": {"domain", "infrastructure"},
    "cli": {"domain", "application", "infrastructure", "datasource", "modeling"},
}


def _airbnb_imports(path: Path) -> set[str]:
    """Return the airbnb layers a module imports (``airbnb.x.y`` -> ``x``)."""
    tree = ast.parse(path.read_text())
    layers: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "airbnb" and len(parts) > 1:
                layers.add(parts[1])
    return layers


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_what_it_may(layer: str) -> None:
    """Each module in ``layer`` imports only itself and the layers it is allowed to see."""
    for module in sorted((PACKAGE / layer).glob("*.py")):
        forbidden = _airbnb_imports(module) - ALLOWED[layer] - {layer}
        assert not forbidden, f"{module.relative_to(PACKAGE)} imports {sorted(forbidden)}"


def test_every_layer_is_covered() -> None:
    """A new top-level package cannot dodge the rule by not being listed."""
    packages = {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    assert packages == set(ALLOWED)
