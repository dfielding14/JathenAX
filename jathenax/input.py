"""Defaults and TOML input decks for the built-in problems."""

import tomllib


PROBLEMS = {
    "smooth": (64, 48, 0.2, "periodic_isothermal.npz"),
    "kh": (128, 128, 3.0, "kelvin_helmholtz.npz"),
    "turbulence": (256, 256, 45.0, "driven_seed1618.npz"),
}


def read_input(path, problem=None):
    """Load a deck over the selected problem's defaults."""
    deck = {}
    if path is not None:
        with path.open("rb") as file:
            deck = tomllib.load(file)

    problem_section = deck.get("problem", {})
    if not isinstance(problem_section, dict):
        raise ValueError("[problem] must be a table")
    name = problem if problem is not None else problem_section.get("name", "smooth")
    if not isinstance(name, str) or name not in PROBLEMS:
        raise ValueError("problem.name must be smooth, kh, or turbulence")
    nx, ny, tlim, filename = PROBLEMS[name]
    driven = name == "turbulence"

    settings = {
        "problem": dict(name=name),
        "mesh": dict(nx=nx, ny=ny),
        "time": dict(tlim=tlim, nlim=100_000, integrator="rk3", cfl_number=0.4),
        "hydro": dict(
            reconstruct="ppm4", rsolver="roe", iso_sound_speed=1.0,
            dfloor=1.17549e-38 if driven else 1e-10,
            viscosity=1.5e-6 if driven else 0.0, smagorinsky_cs=0.0,
        ),
        "hydro_srcterms": dict(drag_rate=0.3333335 if driven else 0.0),
        "sgs": dict(model="", weights=""),
        "output": dict(file=filename),
    }
    if driven:
        settings["turb_driving"] = dict(
            rseed=1618, tcorr=0.75, dt_update=0.0075, tdriv_start=0.0, dedt=0.03703705,
            nlow=3, nhigh=5, npeak=4.0, sol_fraction=1.0,
            min_kx=-5, max_kx=5, min_ky=-5, max_ky=5,
        )

    for section, table in deck.items():
        if section not in settings:
            raise ValueError(f"unsupported input section [{section}] for {name}")
        if not isinstance(table, dict):
            raise ValueError(f"[{section}] must be a table")
        for key, value in table.items():
            if key not in settings[section]:
                raise ValueError(f"unknown input setting {section}.{key}")
            expected = type(settings[section][key])
            types = (int, float) if expected is float else (expected,)
            if type(value) not in types:
                raise ValueError(f"{section}.{key} expects {expected.__name__}")
            settings[section][key] = value
    settings["problem"]["name"] = name
    return settings
