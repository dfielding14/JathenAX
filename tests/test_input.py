from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jathenax.input import read_input


def test_seed1618_deck():
    path = Path(__file__).resolve().parents[1] / "inputs" / "turbulence.toml"
    settings = read_input(path)
    defaults = read_input(None, "turbulence")
    defaults["output"]["file"] = "runs/driven_seed1618.npz"
    assert settings == defaults


@pytest.mark.parametrize(("contents", "message"), [
    ("[hydro]\nviscosty = 0", "unknown input setting hydro.viscosty"),
    ("[mesh]\nnx = true", "mesh.nx expects int"),
    ("[turb_driving]\nrseed = 1", "unsupported input section"),
])
def test_input_rejects_invalid_settings(tmp_path, contents, message):
    path = tmp_path / "input.toml"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_input(path)


def test_input_deck_and_cli_overrides(tmp_path):
    deck = tmp_path / "input.toml"
    deck.write_text("""
[problem]
name = "kh"

[mesh]
nx = 8
ny = 10

[time]
nlim = 2
tlim = 1
integrator = "rk2"

[hydro]
reconstruct = "plm"
rsolver = "hlle"

[output]
file = "configured.npz"
""")
    output = tmp_path / "overridden.npz"
    result = subprocess.run(
        [sys.executable, "-m", "jathenax.run", "-i", str(deck), "--nx", "12",
         "--integrator", "rk4", "--rsolver", "llf", "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "RK4" in result.stdout
    assert "Reconstruction plm, Riemann solver llf" in result.stdout
    with np.load(output) as data:
        assert data["problem"].item() == "kh"
        assert data["conserved"].shape == (10, 12, 4)
        assert data["cycle"] == 2
        assert 0 < data["time"] < 1
    assert not (tmp_path / "configured.npz").exists()
