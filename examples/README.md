# Connect a JAX SGS model

Run the example from the JathenAX checkout. It uses a small constant eddy
viscosity as a stand-in for a trained predictor.

```sh
python -m jathenax.run turbulence --nx 128 --ny 128 --tlim 1 \
    --sgs-model examples.sgs_model --output runs/sgs_example.npz
```

To connect your model, write an importable Python module, for example
`my_sgs_model.py` in the working directory. Use
[sgs_model.py](sgs_model.py) as a template. The module provides two functions.

| Function | Contract |
| --- | --- |
| `load_weights(checkpoint)` | Runs once before integration. Receives a `pathlib.Path` from `--sgs-weights`, or `None`. Return arrays or a JAX pytree of arrays, including any normalization statistics. |
| `predict_stress(weights, primitives)` | Runs inside the JIT-compiled timestep once per RK stage. Receives cell-centered `(rho, vx, vy)` in code units, shape `(ny, nx, 3)`. Return a floating-point array of the same shape with components `(tau_xx, tau_xy, tau_yy)`. |

```sh
python -m jathenax.run turbulence --nx 128 --ny 128 --tlim 1 \
    --sgs-model my_sgs_model --sgs-weights /path/to/checkpoint \
    --output runs/learned_sgs.npz
```

Restore the checkpoint using the same model library and architecture used in
training. Write inference and preprocessing with JAX operations. Apply your
training normalization and any batch or channel transposes inside the
predictor, then convert the output to physical code units and remove any
batch dimension. Select the precision in your adapter. The returned stress
is cast to the solver state dtype before it enters the flux.

JathenAX places the weights on the same device as the flow. The predictor
sees the whole periodic domain, with cell index `[j, i]` corresponding to
`[y, x]`. Apply periodic padding inside the predictor when the model uses
local stencils, including across the edge of the box. A whole-domain model
can use its full receptive field.

## Stress convention

The expected stress is the positive Favre covariance in momentum-flux units,
$\tau_{ij}=\overline{\rho v_i v_j}-\bar\rho\,\tilde v_i\tilde v_j$.
For a model trained to predict covariance per unit mass, multiply by density
in the adapter. JathenAX averages neighboring predictions onto periodic faces
and adds the stress to the momentum flux. The RK update applies the negative
divergence. The isotropic trace is included if the model supplies it.

Smagorinsky is off by default. Enabling it alongside the predictor adds both
stresses. The predictor does not modify density or tracer fluxes directly.
The existing timestep limits cover hydro, explicit viscosity, and drag.
The learned stress supplies no additional stability limit. Use short runs
to check stability before increasing the duration or resolution.

## Check a prediction on a snapshot

This checks the model on the final state from the example above. The same
adapter can read another JathenAX snapshot with `conserved` stored in
`(ny, nx, nvar)` order.

```python
import jax
import jax.numpy as jnp
import numpy as np

from examples.sgs_model import load_weights, predict_stress
from jathenax.hydro import recover_isothermal

with np.load("runs/sgs_example.npz") as snapshot:
    U = jnp.asarray(snapshot["conserved"])
_, primitives = recover_isothermal(U, 1e-10)
weights = jax.device_put(load_weights(None), U.device)
tau = jax.jit(predict_stress)(weights, primitives[..., :3])
assert tau.shape == (*U.shape[:2], 3)
assert bool(jnp.isfinite(tau).all())
```

Python callers can pass the same `predict_stress` and `sgs_weights` directly
to `evolve_isothermal` or `isothermal_step`. Mark the predictor as static
when wrapping a call in your own JIT. Pass the weights as array arguments
so the same compiled function can use different checkpoint values.
