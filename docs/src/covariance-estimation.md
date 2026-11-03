# Covariance estimation

The package includes the covariance-estimation methods described by
[CEDA](https://github.com/ptandeo/CEDA): expectation-maximization (EM) with an
extended Kalman smoother (`em_eks`) and with an ensemble Kalman smoother
(`em_enks`). Both estimate model-error covariance `Q`, observation-error
covariance `R`, and optionally the initial background distribution.
The methods implement the core algorithms from Dreano et al. (2017),
[“Estimating Model-Error Covariances in Nonlinear State-Space Models using
Kalman Smoothing and the Expectation–Maximization Algorithm”](https://doi.org/10.1002/qj.3048).

The observation input is a matrix with variables in rows and time steps in
columns. A `NaN` marks a missing observation; partially observed time steps are
also supported. The dynamics and observation operators receive a state vector.
`em_enks` uses a linear observation matrix, and accepts an explicit random
number generator for reproducibility. The optional `inflation` factor (default
`1`) scales the observation-error covariance used in filter updates; reported
likelihoods use the supplied, uninflated covariance.

```julia
using AnalogDataAssimilation, LinearAlgebra, Random

observations = reshape([0.3, 0.1, NaN, -0.2], 1, :)
f(x) = [0.9x[1]]
jac_f(x) = reshape([0.9], 1, 1)
h(x) = [x[1]]
jac_h(x) = reshape([1.0], 1, 1)

result = em_eks(
    observations;
    initial_state = [0.0],
    initial_covariance = reshape([1.0], 1, 1),
    model_noise_covariance = reshape([0.1], 1, 1),
    observation_noise_covariance = reshape([0.2], 1, 1),
    dynamics = f,
    dynamics_jacobian = jac_f,
    observation_operator = h,
    observation_jacobian = jac_h,
    iterations = 5,
)

Q_estimates = result.model_noise_covariance
R_estimates = result.observation_noise_covariance
```

`CovarianceEMResult` contains smoothed state estimates and their covariance
matrices for each EM iteration, the parameter histories (including the initial
values), and the observed-data log-likelihood. `smoothed_states` is indexed by
variable, time, and EM iteration; `smoothed_covariances` is indexed by row,
column, time, and EM iteration. When a `truth` matrix is supplied, the result
also contains an RMSE per iteration. Model-noise estimates can be unconstrained (`:full`),
diagonal (`:diag` or `:diagonal`), or constrained to a scaled positive-definite
template (`:const` or `:scaled`) using `model_noise_template`.
The `estimate_model_noise`, `estimate_observation_noise`, and
`estimate_background` switches can hold individual parameter groups fixed.
`em_enks` also accepts `ensemble_size` (at least two members).

The executable Literate walkthrough, **CEDA covariance estimation**, is listed
under Examples. It reproduces the Lorenz-63 comparison of EM-EKS and EM-EnKS,
followed by the Lorenz-96 EM-EnKS experiment, including the state/observation
and covariance, likelihood, and RMSE plots.

These methods use a discrete-time transition function. Unlike `StateSpaceModel`,
which integrates a continuous-time ODE, the `dynamics` callback here maps one
state directly to the next. The EKS additionally requires Jacobians for both
the dynamics and observation operator. The translated API covers CEDA's
advertised EM-EKS and EM-EnKS algorithms; the upstream repository's separate
ETKS and adaptive/experimental algorithms are not included.

```@docs
CovarianceEMResult
em_eks
em_enks
```
