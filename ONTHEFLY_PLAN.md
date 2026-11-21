# Learning subgrid transport in 3D turbulent boxes

**JathenAX research and implementation plan · Draft, 21 November 2026**

The experiment asks whether a local neural closure trained through a compressible flow solver can predict the transport and exchange of unresolved kinetic and thermal energy. The science reference is a three-dimensional, continuously driven JathenAX box with an ideal-gas equation of state. Short AthenaK fixtures provide independent numerical checks. The large-eddy simulation (LES) runs on a uniform, triply periodic grid on one GPU. It evolves filtered total energy and subgrid-scale (SGS) kinetic energy using **Form A**.

**Science target.** Train the first-derivative tensor-basis neural network (TBNN) online inside a $64^3$ LES, using a DNS of order $1024^3$ or higher as the reference. At $1024^3$, each coarse cell averages $16^3=4096$ fine cells. Filter the DNS as it runs and save coarse targets at regular physical times together with the driving history. The gradient passes through the coarse solver.

The $32^3\to8^3$ pilot established the training loop. Each online update holds the weights fixed across all Runge–Kutta (RK) stages and compares the result with eight filtered reference states. After 50 updates, the TBNN improved short forecasts on six starts from a second seed. Section 7 records the long-run Mach deficit for comparison as the coarse resolution increases. [Held-out pilot results](validation/README.md#frozen-stress-on-a-held-out-seed)

**First science milestone.** Produce a stationary, single-phase 3D box at a specified Mach number, close its total-energy budget, and measure how much driving work goes directly below the intended filter scale. Repeat the measurement at the filter scale and resolution used for each scientific comparison.

## 1. The coarse system

Use a conservative cubic volume average of width $\Delta$, denoted by an overbar, and the Favre average $\widetilde f=\overline{\rho f}/R$. Let $e$ be fine-grid specific internal energy and define

$$
\begin{aligned}
R &= \overline{\rho}, &
\boldsymbol v &= \widetilde{\boldsymbol u}, &
U &= \overline{\rho e},\\
B &= \tfrac12 R|\boldsymbol v|^2, &
K &= \tfrac12\left(\overline{\rho|\boldsymbol u|^2}-R|\boldsymbol v|^2\right)=Rk, &
\overline E &= U+B+K .
\end{aligned}
$$

Evolve $(R,R\boldsymbol v,\overline E,K)$. Recover the thermal pressure from $P=(\gamma-1)(\overline E-B-K)$. The positive Favre stress is $\tau_{ij}=\overline{\rho u_i u_j}-Rv_i v_j$, so $\operatorname{tr}\boldsymbol\tau=2K$. Four closure quantities enter the equations.

| Model output | Type | Physical meaning |
|---|---|---|
| $\boldsymbol\tau^{\rm dev}$ | Symmetric traceless tensor | Anisotropic momentum transport. Add $(2K/3)\boldsymbol I$ to recover the full stress. |
| $\boldsymbol q$ | Polar vector | SGS enthalpy flux, $R(\widetilde{h\boldsymbol u}-\widetilde h\,\boldsymbol v)$, with $h=e+p/\rho$ on the fine grid. |
| $\boldsymbol F_K$ | Polar vector | Nonadvective SGS kinetic-energy flux, including pressure transport. |
| $\mathcal X$ | Signed scalar | Net thermal-to-SGS exchange. Positive $\mathcal X$ increases $K$ at fixed $\overline E$. |

With external acceleration $\boldsymbol a$ and positive net radiative loss $\overline{\mathcal L}$, the high-Reynolds-number equations are

$$
\begin{aligned}
\partial_t R+\nabla\cdot(R\boldsymbol v)&=0,\\
\partial_t(R\boldsymbol v)+\nabla\cdot
  [R\boldsymbol v\otimes\boldsymbol v+P\boldsymbol I+\boldsymbol\tau]
 &=\overline{\rho\boldsymbol a},\\
\partial_t\overline E+\nabla\cdot
  [(\overline E+P)\boldsymbol v+\boldsymbol\tau\cdot\boldsymbol v+
    \boldsymbol q/\gamma+\boldsymbol F_K]
 &=\overline{\rho\boldsymbol u\cdot\boldsymbol a}-\overline{\mathcal L},\\
\partial_t K+\nabla\cdot[K\boldsymbol v+\boldsymbol F_K]
 &=-\boldsymbol\tau:\nabla\boldsymbol v+\mathcal X+S_{K,\rm drive}.
\end{aligned}
$$

For a constant-$\gamma$ ideal gas, the pressure transport in $\boldsymbol F_K$ is fixed by the measured enthalpy flux. Write $\boldsymbol j=\boldsymbol q_3-\boldsymbol J_\sigma$, where $\boldsymbol q_3=\overline{\rho|\boldsymbol u-\boldsymbol v|^2(\boldsymbol u-\boldsymbol v)}/2$ and $\boldsymbol J_\sigma$ is the viscous-work correlation. Then

$$
\boldsymbol F_K=\boldsymbol j+\frac{\gamma-1}{\gamma}\boldsymbol q,\qquad
\frac{\boldsymbol q}{\gamma}+\boldsymbol F_K=\boldsymbol q+\boldsymbol j .
$$

Use this identity on the faces. A change to the heat model then supplies its share of the flux in both energy equations. The independently modeled part of $\boldsymbol F_K$ is $\boldsymbol j$. The signed exchange $\mathcal X=\mathcal C_p-R\epsilon_\Delta$ combines unresolved pressure work and dissipation. Include resolved molecular momentum and energy fluxes in both the LES and the reference budget wherever viscosity or conduction is significant at $\Delta$.

## 2. The driven, cooled reference box

### Driving above the filter scale

Target $\mathcal M_{\rm box}=0.5$ with continuous, solenoidal, Ornstein–Uhlenbeck acceleration at low Fourier modes and zero linear drag. Use a cube of side $L$ with dimensionless modes $1\leq|\boldsymbol n|\leq3$. On the target $64^3$ coarse grid the shortest driven wavelength spans about 21 cells and the peak wavelength spans 32. Use seed 1618 for matched training runs and reserve other seeds for held-out closure tests.

Start with implicitly dissipative Euler hydro (PPM4, Roe, RK3), with explicit viscosity and thermal conductivity off. This reference measures advective SGS enthalpy flux and direct forcing work without assigning molecular coefficients. The kinetic dissipation near the grid scale depends on resolution and the numerical scheme. Repeat the budget and filtered-moment measurements at the same physical $\Delta=L/64$ before interpreting transport. An independent viscous dissipation label for $\mathcal X$ calls for a resolved constant-viscosity run. A specified Prandtl number also requires explicit thermal conduction. [Schmidt & Grete, forced and cooled compressible ILES](https://doi.org/10.1103/PhysRevE.100.043116) · [Radice et al., resolution dependence in weakly compressible ILES](https://arxiv.org/abs/1501.03169)

Forcing affects a closure in two ways. It sets the population of shocks, compressions, and shear in the training data. It can also do work directly on unresolved velocities. The exact direct source is

$$
S_{K,\rm drive}=\overline{\rho\boldsymbol u\cdot\boldsymbol a}
-\boldsymbol v\cdot\overline{\rho\boldsymbol a}.
$$

It vanishes for an acceleration uniform across the filter volume. Also measure $\overline{\rho\boldsymbol a}-R\boldsymbol a_{\rm coarse}$ using the realized forcing on the same interval. Compare momentum-source errors with the applied impulse and direct SGS work with $-\boldsymbol\tau:\nabla\boldsymbol v$, using signed means and RMS values. If either changes the coarse update appreciably, increase the scale separation or model that source explicitly before an autonomous run. Keep a change of forcing band or solenoidal fraction as a held-out test. [Aluie, *Scale decomposition in compressible turbulence*](https://arxiv.org/abs/1012.5877) · [Federrath et al., *Solenoidal versus compressive turbulence forcing*](https://arxiv.org/abs/0905.1060)

### Cooling and Mach-number calibration

Use $\gamma=5/3$ for the pilot and a constant $\tau_{\rm cool}$ in each run. The fine-grid thermal loss $p/[(\gamma-1)\tau_{\rm cool}]$ filters exactly to $\overline{\mathcal L}=U/\tau_{\rm cool}$. Cooling therefore enters the $\overline E$ equation directly. An isolated cooling update holds $R$, momentum, and $K$ fixed and has the analytic solution $U(t+\delta t)=U(t)\exp(-\delta t/\tau_{\rm cool})$.

Let $\epsilon_{\rm drive}$ be the mean injected power **per unit volume**. With `normalization = edot` the pinned AthenaK driver divides the work sums by box volume before solving for `dedt`. JathenAX uses cell means on its uniform grid. [AthenaK normalization](https://github.com/dfielding14/athenak/blob/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc/src/srcterms/turb_driver.cpp#L1051-L1123) · [JathenAX normalization](jathenax/forcing.py)

In the periodic drag-free box,

$$
\frac{d\langle\overline E\rangle_V}{dt}
=\epsilon_{\rm drive}-\frac{\langle U\rangle_V}{\tau_{\rm cool}},
\qquad
\langle U\rangle_{V,t}=\epsilon_{\rm drive}\tau_{\rm cool}
\quad\text{in statistical steady state}.
$$

Measure kinetic energies in the center-of-mass frame. Define the turbulent Mach number from the ratio of space-time averages,

$$
\mathcal M_{\rm box}^2=\frac{2\langle B+K\rangle_{V,t}}{\gamma\langle P\rangle_{V,t}} .
$$

Including $K$ preserves this diagnostic when the fine field is filtered. For a target $\mathcal M_\star$, fix the driving band and power, measure $\langle B+K\rangle$ in a pilot, and estimate

$$
\tau_{\rm cool}^{(0)}
=\frac{2\langle B+K\rangle_{\rm pilot}}
{\gamma(\gamma-1)\epsilon_{\rm drive}\mathcal M_\star^2}.
$$

Once the box is stationary, this update is approximately $\tau_{\rm new}=\tau_{\rm old}(\mathcal M_{\rm measured}/0.5)^2$. Rerun and remeasure because the saturated kinetic energy can change. Compare kinetic and thermal energy in successive time blocks and check the box energy storage against injection minus cooling. Estimate Mach uncertainty from paired energy blocks longer than their measured correlation times. Keep the target fixed when transferring the calibration to the science grid. [Stationary calibration procedure](validation/README.md#calibrate-the-stationary-mach-number)

For the unit box with $\epsilon_{\rm drive}=0.03703705$, the $32^3$ AthenaK calibration at $\tau_{\rm cool}=14$ gave $\mathcal M_{\rm box}=0.4974$ and $0.4977$ for seeds 1618 and 1619 over $60\le t\le120$. The first native $32^3$ pilot used unit driving power, $\tau_{\rm cool}=14/3$, $t_{\rm corr}=0.25$, a driving update interval of 0.0025, and initial sound speed 3. Seed 1618 gave $\mathcal M_{\rm box}=0.4978\pm0.0017$ and mean cooling divided by injection of 1.00192 over $20\le t\le40$.

Set the OU correlation time for science runs to the turnover time at the peak forcing wavelength, $t_{\rm eddy}=(L_{\rm box}/n_{\rm peak})/v_{\rm rms}$, using the stationary 3D mass-weighted rms turbulent speed. The first pilot gave $t_{\rm eddy}=0.441$ for $n_{\rm peak}=2$. With $t_{\rm corr}=0.44$, the revised $32^3$ pilot gave $t_{\rm eddy}=0.43985$ and $\mathcal M_{\rm box}=0.4989\pm0.0023$. Measure the kinetic energy and check $t_{\rm corr}/t_{\rm eddy}$ again at production resolution. Reserve fresh seeds for closure evaluation.

Starting the same $32^3$ driver with $c_{s,0}=2.28$ and a fresh solenoidal velocity field with $v_{\rm rms,0}=1.14$ brought cooling within 2% of injection after one turnover. Both energy balance and the velocity spectrum passed for three consecutive turnovers by $7t_{\rm eddy}$, compared with $42t_{\rm eddy}$ from rest at $c_{s,0}=3$. This spin-up estimate applies to one realization at $32^3$. [Paired spin-up measurements](validation/README.md#measure-spectral-spin-up)

## 3. Hydro and forcing implementation

JathenAX advances periodic gas dynamics in one, two, and three dimensions. The ideal-gas state includes total energy, three-component forcing in 3D, and stage-wise thermal cooling. The hydro step applies learned stress in 2D isothermal runs. The 3D Form A pilot evolves $K$, includes its isotropic pressure work, and evaluates the TBNN on the current coarse state at each RK stage. [Gas hydro step](jathenax/hydro.py) · [3D LES step](jathenax/les.py) · [3D stress predictor](jathenax/tbnn.py) · [Face operators](jathenax/fluxes.py)

1. **Establish 3D gas dynamics with $K=0$.** Start with PPM4, the AthenaK adiabatic Roe solver, and RK3. Compare smooth waves along each axis, a shock tube, and intermediate RK states against a matched AthenaK build in float64. Reuse RK2 and RK4 through the same stage interface and check the existing 2D regressions.
2. **Port the three-component forcing and cooling.** Match the mode order, RNG draws, projection, forcing refresh times, and finite kick. For a resolved velocity impulse $\delta\boldsymbol v$, add $\boldsymbol m_{\rm before}\cdot\delta\boldsymbol v+R|\delta\boldsymbol v|^2/2$ to total energy. Verify that the kick alone preserves thermal energy cell by cell. Check the isolated cooling solution and convergence of a combined driven, cooled run. [AthenaK forcing reference](https://github.com/dfielding14/athenak/blob/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc/src/srcterms/turb_driver.cpp)
3. **Solve the isotropic subsystem inside the Riemann solver.** Set $P_t=2K/3$ and $P_{\rm eff}=P+P_t$. For adiabatic isotropic compression, $c_{\rm eff}^2=(\gamma P+5P_t/3)/R$. At $\gamma=5/3$, $P_{\rm eff}=2(\overline E-B)/3$. Use the ordinary Euler momentum and total-energy fluxes with this pressure and its acoustic speed. They include isotropic stress work and advection of the energy in $K$. In the $K$ equation advect $K/R$ with the face mass flux and use a face velocity consistent with the pressure work for $-P_t\nabla\cdot\boldsymbol v$. Test a nonzero-$K$ acoustic wave and verify material conservation of $K/R^{5/3}$ in smooth isotropic adiabatic compression. Repartitioning $U+K$ at fixed $R,\boldsymbol v,\overline E$ leaves $P_{\rm eff}$ invariant and changes the cooling rate. Use thermal $P$ for temperature, Mach number, and cooling. Defer other $\gamma$ until this split passes. [Semenov, discussion of thermal and turbulent pressure](https://arxiv.org/html/2410.23339v3#S2)
4. **Couple the remaining face fluxes.** Add the deviatoric stress to the Euler momentum flux and its work to total energy. For a cell-centered velocity $\boldsymbol v_c$, face velocity $\boldsymbol v_f$, and face divergence $D_h$, use $\boldsymbol v_c\cdot D_h\boldsymbol\tau_f^{\rm dev}-D_h(\boldsymbol\tau_f^{\rm dev}\cdot\boldsymbol v_f)$ for the corresponding semi-discrete $K$ production. Share each face's $\boldsymbol q$ and $\boldsymbol j$ between the two energy equations using Section 1. Check periodic conservation, exchange at fixed total energy, and convergence of the transfer under timestep refinement.

**Admissibility during training and production.** For an isolated exchange substep, bound $\delta K=\delta t\,\mathcal X$ to $[-K,U-U_{\min}]$ and hold $\overline E$ fixed. For a positive filter the full Favre stress must have trace $2K$ and nonnegative eigenvalues. If necessary, scale the traceless part toward zero just enough to satisfy this bound before using the same stress for flux and work. This preserves the possibility of either sign of stress production. Measure the unmodified minimum eigenvalue and the fraction and magnitude of corrections, including for the analytic baseline.

Choose the timestep from the acoustic, explicit transport, and source rates. The current 2D stress hook supplies no additional stability bound. For a learned gradient-dependent flux, check the rate implied by its response to a gradient perturbation and verify timestep convergence with frozen weights. Check $R>0$, $K\geq0$, $U\geq U_{\min}$ and finite fields after each transport/source stage. In production, restore the last valid state and retry an invalid step at smaller $\delta t$ using the same forcing realization. Stop and record the failing state if refinement fails. In training, reject a weight update that fails the prescribed window and retry that window with a smaller optimizer step. If the baseline itself fails, refine the common schedule or repair the flux before comparing models. Report limiter activity and floor corrections against the measured physical transfers. Frequent corrections invalidate an inference about the unmodified closure. [Current timestep limits](jathenax/hydro.py)

Use the existing TOML deck convention for the EOS, third grid dimension, forcing, cooling, and closure selection. Record the resolved parameters and code revisions with each run. Use the same coarse RK update and stability limits in training and production.

## 4. What the reference simulation must save

Use aligned coarse cells and accumulate the reference quantities on the fly. Save short bursts for training and replay, plus lower-cadence histories for long-run statistics. At the pinned revision, AthenaK's `hydro_sgs_3d` block-averages density and momentum moments and outputs $R$, $\boldsymbol v$, and the six components of $\boldsymbol\tau$. This supplies $K=\operatorname{tr}\boldsymbol\tau/2$. The standard coarsened binary writer casts fields to float32. Use float64 for a few diagnostic bursts to resolve small differences of large energy terms, and measure the effect of downcasting before choosing the training output precision. [Coarsened moments](https://github.com/dfielding14/athenak/blob/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc/src/outputs/coarsened_binary.cpp#L392-L420) · [Binary field precision](https://github.com/dfielding14/athenak/blob/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc/src/outputs/coarsened_binary.cpp#L712-L725)

| Reference quantity | Use |
|---|---|
| Filtered total and thermal energy, density, momentum, and stress | Initialize the LES from a developed state, recover $P$ and $K$, and verify $\overline E=U+B+K$ independently. |
| Filtered enthalpy transport and central third velocity moments | Construct $\boldsymbol q$ and $\boldsymbol F_K$ in the convention of Section 1. Include the viscous-work correlation for an explicitly viscous reference when significant. |
| Pressure work and dissipation measured independently where possible | Construct a physical $\mathcal X$ label with an uncertainty estimate. Keep a $K$-budget remainder as a separately named diagnostic. It inherits errors in transport, stress production, and implicit dissipation. |
| Absolute kick times, filtered momentum impulse and work in each coarse cell, pre-kick state, forcing coefficients and normalization | Reproduce the finite kicks and quantify direct SGS injection. A seed reproduces the raw pattern. The amplitude also depends on the flow being kicked. |
| RK-weighted, time-integrated fine numerical mass, momentum, and total-energy fluxes crossing each coarse face, plus volume-integrated sources and floor corrections | Recover each coarse cell's conserved increment. Accumulate on aligned faces for a few validation bursts. Routine training needs only the targets and forcing history. |

For a single fine kick let $\boldsymbol I=\overline{\rho\,\delta\boldsymbol u}$ and $W=\overline{\rho\boldsymbol u^-\cdot\delta\boldsymbol u+\rho|\delta\boldsymbol u|^2/2}$ be the restricted momentum and energy impulses per unit volume. At the reference state the exact split is

$$
\delta B=\boldsymbol v^-\cdot\boldsymbol I+\frac{|\boldsymbol I|^2}{2R},
\qquad \delta K_{\rm drive}=W-\delta B .
$$

Accumulate this split at the individual kicks, preserving their timing relative to the hydro update. For matched training windows prescribe the reference impulse history. Compute $\delta B$ on the evolving coarse state, add the recorded $\delta K_{\rm drive}$ to $K$, and add their sum to $\overline E$. This preserves $U$ in a kick and reproduces $W$ on the reference state. Check convergence of any time aggregation. Autonomous tests use the production power normalization and the forcing treatment justified in Section 2.

### Replay measured closures before trajectory training

1. **Close the reference update.** For every coarse cell, compare the restricted fine change in $(R,R\boldsymbol v,\overline E)$ with the oriented, area- and time-integrated fine flux imbalance plus all sources, divided by cell volume. Agreement should reach the precision of the stored sums. $K$ is derived from the filtered second moment. Compute its two endpoints independently. The fine Euler solver has no conserved $K$ face flux to sum.
2. **Run the actual coarse operator.** Start from the same restricted state $Q_0=(R,R\boldsymbol v,\overline E,K)_0$. Prescribe measured $\boldsymbol\tau^{\rm dev},\boldsymbol q,\boldsymbol j$ and independently estimated $\mathcal X$ at the coarse RK stage times. Keep the production face interpolation, Riemann solver, source ordering, and matched impulses. For one step, then a short window, measure $\mathcal R_\Delta=[Q^{\rm ref}(t_1)-\Phi_\Delta(Q_0)]/(t_1-t_0)$, where $\Phi_\Delta$ is this coarse update. Tighten the moment sampling cadence and timestep to quantify interpolation and time-integration errors. If $\mathcal X$ is available only as a budget remainder, report the $K$ replay conditional on that estimate. Reusing the same budget to construct and validate $\mathcal X$ is circular.
3. **Interpret the residual.** Compare each residual with the SGS contribution in the same units and at the same spatial scale. Estimate uncertainty from output precision, sampling cadence, and fine-grid refinement at fixed $\Delta$. The instantaneous mass flux at a prescribed Favre state has no independent SGS term, so it provides a check on discretization and filtering. A closure that offsets numerical diffusion and improves held-out JathenAX trajectories meets a prediction objective for that discretization. Assess physical transport against independent moments. Use the same windows with a second coarse flux choice to measure sensitivity to the numerical solver, and test frozen weights on the target solver before claiming transfer. Agdestein & Sanderse demonstrate the role of discretization in incompressible LES. Measure its size here for compressible flow. [Agdestein & Sanderse](https://arxiv.org/abs/2403.18088)

## 5. The local tensor/vector-basis model

A single small multilayer perceptron (MLP) learns scalar coefficients. The features come from the current **coarse** state. On the cubic grid, use the 48 signed permutations of the axes, the full cubic group $O_h$. Its 2D counterpart is the eight-element square group $D_4$. Scalar invariants are unchanged by each symmetry, a polar vector transforms as $Q\boldsymbol F$, and a stress transforms as $Q\boldsymbol\tau Q^{\mathsf T}$ at the transformed location. [Ling et al., *Deep neural networks with embedded invariance*](https://doi.org/10.1017/jfm.2016.615)

Start with compact periodic **first derivatives only**. Write $G_{ij}=\Delta\,\partial_j v_i/c_s$, $\boldsymbol g_h=\Delta\nabla\widetilde h/c_s^2$ and $\boldsymbol g_k=\Delta\nabla k/c_s^2$, where $c_s^2=\gamma P/R$ and $\widetilde h=\gamma U/R$. A small pilot basis uses the traceless parts of $(G+G^{\mathsf T})/2$ and $GG^{\mathsf T}$ for stress and $\{\boldsymbol g_h,G\boldsymbol g_h,\boldsymbol g_k\}$ for the vector flux. The $G\boldsymbol g_h$ term allows heat transport across the mean enthalpy gradient and has the tensor structure of the leading smooth-field box-filter covariance. Scalar inputs can start with $k/c_s^2$, dilatation, strain and rotation norms, and the norms of $\Delta\nabla\ln R$, $\boldsymbol g_h$, and $\boldsymbol g_k$. Fix the list, ordering, stencils and normalization for all pilot controls.

For the stress-only pilot, activate the stress correction and keep the other closures analytic. For the subsequent stress-and-heat comparison, initialize both learned outputs as corrections to the corresponding analytic closures. With $v_K$ a speed that approaches $\sqrt{k}$ in developed turbulence and vanishes at $k=0$, use

$$
\begin{aligned}
\boldsymbol\tau^{\rm dev}_\theta &= \boldsymbol\tau^{\rm dev}_{\rm an}
       +2K\sum_a A_a(\mathcal I)\boldsymbol T_a,\\
\boldsymbol q_\theta &= \boldsymbol q_{\rm an}
       +R c_s^2 v_K\sum_b H_b(\mathcal I)\boldsymbol V_b .
\end{aligned}
$$

Set the MLP's output corrections to zero initially. A positive filter gives $|\boldsymbol q|^2\leq 2RK\,\widetilde{(h-\widetilde h)^2}$ by Cauchy–Schwarz. At $K=0$ the advective enthalpy and third-moment fluxes vanish even when thermal variance is finite. Apply this limit to the analytic terms and to later transport corrections. For differentiability at zero one choice is $v_K=k/(\sqrt{k+k_0}+\sqrt{k_0})$, with a small positive $k_0$ in specific kinetic-energy units. Vary $k_0$ below the reliably measured range and check the low-$K$ bins. The bound gives an endpoint constraint and leaves the scaling in developed flow to the data.

For the later fits, the same MLP can add a scalar correction $(Rc_s^3/\Delta)D(\mathcal I)$ to $\mathcal X$ and a vector correction $Rc_s^2v_K\sum_b J_b(\mathcal I)\boldsymbol V_b$ to $\boldsymbol j$. Test the entire predictor, including the periodic seam, under all 48 cubic symmetries and a uniform velocity boost. Existing supervised weights are a candidate initialization if their dimensionality, filter, units, basis, and stress sign match. Use the matched supervised checkpoint in Section 6 for the controlled comparison. Add a second- or third-derivative basis only if validation residuals suggest missing spatial structure. Check timestep sensitivity and cost each time the stencil grows.

## 6. Training through the solver

**First test.** Use the existing 2D example stress with its single viscosity coefficient. Generate a smooth target using exactly the same predictor and a known float64 coefficient. The example takes cell-centered gradients, while the built-in viscosity uses face gradients, so those two operators would define a different inverse problem. Check the one-step loss gradient against centered finite differences over a range of perturbations, then recover the coefficient from short trajectories. PPM limiting and Roe switches make the step piecewise differentiable. Repeat the gradient check on representative 3D states. [Example predictor](examples/sgs_model.py) · [Built-in viscous flux](jathenax/fluxes.py)

A CPU float64 pilot on the built-in $12^2$ smooth initial state recovered $\nu=0.0018$ from an initial guess of $0.0008$ using the mean-square conserved-field error over four PPM4/Roe/RK3 steps of $\delta t=0.012$. At the initial guess the rollout gradient and a centered difference with $\delta\nu=10^{-7}$ agreed to $3.6\times10^{-11}$ relative. This check used a synthetic target generated with the same SGS operator.

**One weight update.** Keep the fine reference, labels, forcing history, and prescribed coarse timesteps fixed with respect to the weights. Differentiate through the coarse RK steps over a duration $T=t_N-t_0$ spanning roughly 4–8 coarse CFL steps in the pilot. Record $T/t_\Delta$ with $t_\Delta=\Delta/\sqrt{\langle K\rangle/\langle R\rangle}$ from the developed reference. Use `jax.lax.scan` around the shared RK step. Verify the forward trajectory at identical prescribed timesteps and forcing, then exercise the production adaptive `while_loop` with frozen weights. Use the same optimization for saved and live reference data. [JAX control-flow differentiation](https://docs.jax.dev/en/latest/control-flow.html)

**Comparison times and solver timesteps.** Compare the two solutions at the same physical times $t_n$. These may be evenly or unevenly spaced, and each solver can take smaller stable steps to reach them. A closure trained at one coarse timestep may compensate for RK3 truncation error at that step size. Kochkov et al. explicitly allowed their models to specialize to a fixed timestep. [Kochkov et al., Appendix B](https://arxiv.org/pdf/2102.01010#page=9)

Train on physical fields $z=(R,\boldsymbol v,U,K)$, with scales $s_a$ fixed from the training set. To estimate a time average, weight each endpoint error by the fraction of elapsed time it represents. Even spacing gives each of the $N$ endpoints weight $1/N$.

$$
\mathcal J(\theta)=
\sum_{n=1}^{N}\frac{t_n-t_{n-1}}{T}\sum_a w_a
\left\langle\frac{\lVert z^\theta_a(t_n)-z^{\rm ref}_a(t_n)\rVert^2}{s_a^2}\right\rangle_V
+\lambda_{\rm SGS}\mathcal J_{\rm SGS}.
$$

**What trajectories identify.** For any sufficiently regular periodic vector field $\boldsymbol G$, the transformation

$$
\boldsymbol F_K'=\boldsymbol F_K+\boldsymbol G,\qquad
\boldsymbol q'=\boldsymbol q-\gamma\boldsymbol G,\qquad
\mathcal X'=\mathcal X+\nabla\cdot\boldsymbol G
$$

preserves both energy updates. The total flux $\boldsymbol q/\gamma+\boldsymbol F_K$ is invariant, and the added $K$-flux divergence cancels the added exchange. The same cancellation holds discretely with the same face field and $D_h$ in both places. The basis and admissibility constraints restrict which transformations a model can represent. Individual heat-flux or exchange claims still require independent evidence. Evaluate $\mathcal J_{\rm SGS}$ on reference inputs, using direct labels for each active output with a defensible measurement. Reserve budget-derived exchange for a separately reported effective target. Test each output on held-out reference moments as well as scoring the trajectory.

Split by seed, or by time blocks separated by the measured decorrelation time, before making overlapping windows. Fit normalization on the training split. Select loss weights and window length on validation data, then freeze them for the held-out comparisons.

**Single-GPU LES training.** Keep the coarse flow, model weights, optimizer state, and current filtered targets on the same device. The fine DNS can run separately and supply the same filtered records. Evaluate spatial features from the evolving state once per RK stage and reuse them across heads. Compile the rollout, gradient, and optimizer update together with dynamic weight arrays and fixed window shapes. Measure steady-state training windows per second, peak device memory, and the cost of a frozen closure per hydro step. If storing backward-pass intermediates limits the window, benchmark checkpointing the step body. Save weights, feature order, units, and normalization in a checkpoint loadable by the ordinary production runner.

### Controlled closure experiments

Use eddy viscosity for the analytic deviatoric stress and gradient diffusion of $\widetilde h$ for the enthalpy flux $\boldsymbol q_0=\boldsymbol q_{\rm an}$, with diffusivities proportional to $\Delta v_K$. Keep the diffusion law $\boldsymbol F_{K,0}=-R C_D\Delta v_K\nabla k$ on the **full** SGS kinetic-energy flux. Both neural pilot runs at $\gamma=5/3$ use the same analytic remainder, evaluated from each incoming RK state. With the common face interpolation the fluxes satisfy

$$
\boldsymbol j_0=\boldsymbol F_{K,0}-\tfrac{2}{5}\boldsymbol q_0,\qquad
\boldsymbol F_{K,\theta}=\boldsymbol j_0+\tfrac{2}{5}\boldsymbol q_\theta
=\boldsymbol F_{K,0}+\tfrac{2}{5}(\boldsymbol q_\theta-\boldsymbol q_0).
$$

At fixed stress, implement the heat correction by adding $\boldsymbol q_\theta-\boldsymbol q_0$ to the baseline total-energy face flux and $2/5$ of that correction to the baseline $K$ face flux. A zero heat correction then recovers both exactly. Use the same coefficients and Section 5 low-$K$ prescription wherever those closures are active. Check the implied $\boldsymbol j_0$ against the measured third-moment and viscous-work remainder. Keep $\mathcal X_{\rm an}=-R C_\epsilon k^{3/2}/\Delta$ for exchange. Fit analytic coefficients once on training data. For the two neural pilot runs, fork the same supervised checkpoint. Spend the same number of further updates on the same reference windows, with the corresponding objective. Report both training costs.

| Run | Fitted terms and objective | Purpose |
|---|---|---|
| Ordinary coarse hydro | Gas-only Euler with its native numerical dissipation | Measures what the extra SGS state and closures buy over standard coarse hydro. |
| Analytic Form A | All four quantities analytic | Controls for the extra state and its pressure, transport, and cooling. |
| Supervised stress and heat | $\boldsymbol\tau^{\rm dev},\boldsymbol q$ fitted to moment labels alone | Tests direct supervision with the same architecture, labels, and analytic $\boldsymbol j,\mathcal X$. |
| Rollout-trained stress and heat | Same two outputs, moment and trajectory losses | Isolates the contribution of differentiating through the solver. |
| Add local exchange | Also fit $\mathcal X$ | Test after its physical label or explicitly solver-dependent target is established. |
| Full closure | Also fit $\boldsymbol j$ and reconstruct $\boldsymbol F_K$ | Tests the remaining nonadvective transport on held-out references. |

For gas-only hydro, initialize $R$, $R\boldsymbol v$, and $E_{\rm gas}=\overline E$. At $\gamma=5/3$ this matches the initial supporting pressure $P+2K/3$, assigns $K$ to heat, and increases the initial cooling loss by $K/\tau_{\rm cool}$. Also check short windows with $E_{\rm gas}=B+U$, which matches the initial thermal loss and gives supporting pressure $P$. Let the long runs equilibrate independently at the same driving power and cooling time. With negligible floor corrections, the stationary box-averaged loss equals the injected power. Test the physical effect of storing energy in $K$ through the timing and spatial distribution of cooling and the partition between $U$ and $K$. Use the shared Form A state for attribution between closures.

**How much does the heat flux change thermal energy?** At a common stage state with stress and $\boldsymbol j$ fixed, the heat flux enters total energy as $\boldsymbol q_f$ and $K$ as $2\boldsymbol q_f/5$. Its direct thermal tendency and the measured-closure replay residual in the same budget are

$$
H(\boldsymbol q_f)=-D_h(\boldsymbol q_f/\gamma)=-\tfrac{3}{5} D_h\boldsymbol q_f,\qquad
\mathcal R_U=\frac{\Delta U^{\rm ref}-\Delta U^{\rm replay}}{t_1-t_0}.
$$

Use the filtered fine thermal energy for $\Delta U^{\rm ref}$. For $\Delta U^{\rm replay}$, difference $\overline E-B-K$ evaluated at the two coarse endpoints. Compare the RK-weighted window mean of $H(\boldsymbol q_f^{\rm ref})$ with $\mathcal R_U$ cell by cell, then report RMS by spatial scale and uncertainty across paired windows. Also evaluate $H(\boldsymbol q_f^{\rm ref}-\boldsymbol q_{0,f})$ on the same reference states to measure the thermal tendency the analytic model misses. A spatially constant heat flux can be large and give zero thermal tendency. If $\mathcal X$ comes from the same $K$ budget, its uncertainty also enters $\mathcal R_U$.

On held-out windows, swap the heat closure at fixed stress-model weights and analytic remainder to measure the trajectory response, including feedback on the flow. Score the vector flux against measured moments and compare the resolved temperature variance or spectrum. If the stationary box has too little thermal signal, add a small smooth pressure-balanced entropy perturbation to developed fine turbulence. Re-filter that run and its unperturbed partner, and compare perturbation decay under the same forcing history.

## 7. Milestones and acceptance tests

| Milestone | Evidence to produce |
|---|---|
| 1. Differentiate the existing step | Demonstrated for the smooth 2D scalar-viscosity pilot in Section 6. The 3D closure gradient check belongs to milestone 4. |
| 2. Establish the 3D reference box | Gas-only parity, nonzero-$K$ isotropic compression and wave tests in the LES, finite-kick and cooling-only tests, then a stationary reference with measured Mach number and injection-minus-cooling balance. |
| 3. Replay measured closures before trajectory training | Verify the restricted fine conservative update, quantify direct forcing and coarse-operator residuals, and resolve the thermal signal against uncertainty. Repeat diagnostic bursts at finer reference resolution and cadence. Record which physical terms the data can test. |
| 4. Validate the model and training path | Verify analytic initialization, cubic symmetry, boost invariance, low-$K$ behavior, the 3D trajectory gradient, forward integration, and a saved-checkpoint production run. Freeze the TBNN weights and halve each coarse timestep for both the TBNN and analytic Form A, using the same initial state, target times, and recorded driving kicks. If refinement substantially changes their relative trajectory errors, test training across several stable coarse timesteps. Report training throughput, memory, failure rates, and limiter activity. |
| 5. Compare frozen closures | Run both non-neural controls and the supervised and rollout pair first. On held-out seeds report direct moment and thermal-structure errors, kinetic-energy spectra, Mach number, and $K$ and $U$ statistics over many turnover times. Track mass, energy residuals, transfer, positivity, and timestep sensitivity. Add the exchange and full-transport runs when the labels support the intended claims. |
| 6. Test the physical range | Change forcing band or mixture, target Mach number, and cooling time in separate experiments. Compare with filtered references at matched conditions and estimate statistical scatter across seeds or independent time blocks. |

The first reference accounting test passed on a single periodic $32^3$ AthenaK block filtered to $8^3$. Over four consecutive driven, cooled steps during spin-up, the largest one-step cell residuals were $1.05\times10^{-15}$ in density, $1.53\times10^{-16}$ in momentum density, and $8.53\times10^{-16}$ in total-energy density (upper bounds in code units). A cooling-only run reproduced the discrete RK3 solution. The same check passed on four steps from the developed $32^3$ flow at $t=120$. The $8^3$ JathenAX Form A pilot now replays measured deviatoric stress. A stationary higher-resolution reference and replay with independently measured energy exchange remain open. [Reproduce the reference window](validation/README.md#close-a-3d-athenak-reference-window)

The native gas-only solver passes axis-aligned acoustic waves, a periodic Sod shock tube, isolated cooling, and finite-kick energy checks. Its 3D smooth and velocity-jump conserved states agree with the pinned AthenaK build to $6.3\times10^{-15}$ at all stages of one RK2, RK3 or RK4 step. Native $32^3$ evolution from rest, restricted to $8^3$ at the end of each of four driven, cooled spin-up steps, agrees with the archived conserved fields to $1.8\times10^{-15}$. [Native validation](validation/README.md#verify-native-periodic-hydro)

### Implementation status at $32^3\to8^3$

1. **Record the filtered reference.** The seed-1618 stationary checkpoint at $t=40$ supplies the start state. The native writer supports direct 4:1 volume and face restriction. The 32 saved fine steps end at $t=40.083575$ and span about 6.9 coarse CFL steps at the wave speeds of the filtered states. The float64 records close the per-cell total-energy increment to $7.8\times10^{-15}$ per step. Initially, 34.1% of turbulent kinetic energy is below the coarse scale. Across the 32 steps, 18.6% of the injected work goes directly into $K$. Preserve the measured impulse and its resolved/SGS work split when matching the fine trajectory. [Reference format and checker](validation/README.md#gpu-timing-and-replay-windows)
2. **Fit the first-derivative stress head to the saved states.** The fit used the two stress tensors in Section 5, with zero initial correction to the calibrated eddy viscosity. On the final eight fine steps the relative stress RMSE fell from 0.976 for eddy viscosity to 0.759 with constant tensor coefficients and 0.735 with the TBNN. A fresh GPU process reproduced every prediction bit for bit. All 48 cubic symmetries, a uniform boost, the zero-$K$ limit, and a weight-gradient finite difference passed. Training and evaluation share one short realization. [Reproduce the supervised fit](validation/README.md#supervised-stress-fit)
3. **Replay the $8^3$ Form A update — complete.** The coarse solver took all 32 kicks at their recorded times and scored eight comparison intervals of 0.00806–0.01101 code time units. With common analytic heat, kinetic transport, and exchange models, the normalized trajectory MSE was 0.22821 using measured stress, 0.25172 using eddy viscosity, and 0.24245 using the supervised TBNN. Halving the hydrodynamic timestep changed the latter two scores by less than $10^{-5}$. The measured-stress residual includes errors from the coarse discretization and the analytic energy terms.
4. **Train through the saved trajectory and continue with live data — complete.** The first optimizer step reduced the TBNN trajectory MSE to 0.24226. A fresh fine run regenerated the archived update. Writing and rereading identical live inputs reproduced the loss, gradient, and Adam state bit for bit on each backend. Two more updates used fresh targets every 0.01 code time units through $t=40.24$, carrying the fine state, weights, and optimizer moments forward. Each update reduced the error on its trajectory. Eddy viscosity gave a slightly lower error on the final trajectory. CPU float64 and Metal float32 both passed finite-difference, conservation, and frozen-checkpoint reload checks. [Reproduce the online pilot](validation/README.md#online-stress-training)
5. **Validate frozen weights, then increase resolution.** Fifty updates on seed 1618 reduced mean trajectory MSE from 0.22736 to 0.22534 relative to supervised pretraining on six held-out seed-1619 starts. With one initialization and matched driving impulses over 5.65 turnovers, the learned run conserved energy and gave mean $K=0.2185$ against 0.2242 in the reference. Its resolved kinetic energy was 0.2674 against 0.4095 and Mach was 0.4256 against 0.4937. The analytic model also lost resolved kinetic energy. The peak forcing wavelength spans only four cells at $8^3$. Carry Mach and the kinetic-energy partition forward as resolution diagnostics and proceed to the larger grids. [Reproduce the frozen comparisons](validation/README.md#frozen-stress-on-a-held-out-seed)

**Resolution sequence.** The first $128^3\to32^3$ GPU replay used the held-out seed-1619 checkpoint after five turnovers, with eight filtered targets from $t=2.20$ through $2.27$. On this one trajectory the normalized MSE was 0.13397 for analytic stress, 0.12964 for the pretrained TBNN and 0.12798 for the frozen 50-update TBNN. Starting from the 50-update checkpoint, ten online updates at $32^3$ on seed 1618 reduced the same held-out MSE to 0.12745. The additional gain was 0.41% and the relative stress RMSE rose slightly from 0.73416 to 0.73492. The median update with compiled kernels took 87 seconds on the M4 Max GPU. This comparison covers one short held-out trajectory. [Reproduce the larger-grid training and comparison](validation/README.md#fine-tuning-the-stress-model-at-higher-resolution)

A $256^3\to64^3$ pilot can exercise the target coarse solver and training cost. For the science run, hold the LES at $64^3$ and the physical filter width at $\Delta=L/64$ while increasing the reference to $1024^3$ or higher. Recheck the filtered stresses, energy partition and direct SGS driving work at that width. For a DNS, specify the physical transport and verify that the fine grid resolves dissipation.

Standalone forcing, the gas-only control and held-out seeds beyond 1619 remain open.

## 8. Extension to a thermally bistable medium

Replace the linear loss in the **total-energy source** with a specified net heating/cooling law, for example $\mathcal L(n,T)=n^2\Lambda(T)-n\Gamma$, with stable warm and cold branches over the intended pressure range. For each fine run, output the true filtered net loss and compare it with the value computed from coarse density and temperature. Cells with the same coarse state can contain different mixtures of warm and cold gas and have different radiative losses.

Choose a cold-phase temperature threshold $T_c$ for this cooling law before extracting labels. With $\chi_c=\mathbf 1_{T<T_c}$, the prognostic variable is **cold mass density** $Rf_c=\overline{\rho\chi_c}$. Define the cold-mass flux relative to the bulk flow and the signed rate of phase conversion by

$$
\partial_t(Rf_c)+\nabla\cdot(Rf_c\boldsymbol v+\boldsymbol F_c)=\Gamma_c,\qquad
\boldsymbol F_c=\overline{\rho\chi_c\boldsymbol u}-Rf_c\boldsymbol v,\qquad
\Gamma_c=\overline{\rho D_t\chi_c}.
$$

Measure $\boldsymbol F_c$ from the fine field and estimate conversion from threshold crossings or a time-integrated cold-mass budget with measured transport. For the first multiphase run use $\boldsymbol F_c=-R C_c\Delta v_K\nabla f_c$, calibrate the diffusivity and a local conversion law on training data, and preserve $0\leq Rf_c\leq R$ with a bounded conservative transport/source update. Conversion assigns equal and opposite mass to the complementary phase. Model the radiative correction $\overline{\mathcal L}-\mathcal L(n_\Delta,T_\Delta)$ relative to the loss evaluated at coarse number density and thermal temperature. Include it once in $\overline E$. The exchange $\mathcal X$ only repartitions energy.

Test the cooling closure at matched $R,U,K,\boldsymbol v$, with the other coarse fields alone, then with measured $f_c$, then with evolved $f_c$. Include different initial phase fractions and transients. Check that $f_c$ actually varies at comparable coarse states. If measured $f_c$ adds little to held-out cooling accuracy across those conditions, reconsider its use in the radiative closure over that range. If measured $f_c$ helps and evolved $f_c$ loses the gain, compare the predicted cold-mass flux and conversion with their reference budgets.

Make $f_c$ available to the phase-transport and conversion closures. Assess the value of evolving $Rf_c$ separately through future cold-mass changes, phase conversion, and cold-mass histories, alongside the pressure–density distribution and integrated heating/cooling. Check threshold sensitivity and convergence of interfaces with the stated physical conduction or transport prescription. This extension follows the single-phase tests. [Koyama & Inutsuka, *The Field condition*](https://arxiv.org/abs/astro-ph/0302126)
