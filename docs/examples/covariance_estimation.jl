# # CEDA covariance estimation
#
# This example translates the experiments in the upstream
# [`test_CEDA.ipynb`](https://github.com/ptandeo/CEDA/blob/master/test_CEDA.ipynb)
# notebook into Julia. CEDA estimates model-error covariance ``Q`` and
# observation-error covariance ``R`` with expectation-maximization (EM), using
# either the extended Kalman smoother (EM-EKS) or the ensemble Kalman smoother
# (EM-EnKS). The experiments below retain both upstream models, observation
# schedules, initial covariance guesses, ensemble size, and iteration counts.
#
# ```math
# x_k = f(x_{k-1}) + \eta_k,\qquad
# y_k = h(x_k) + \epsilon_k,
# ```
#
# where the independent, zero-mean Gaussian errors have covariances ``Q`` and
# ``R``. Estimating these covariances with Kalman smoothing and EM is described
# by Dreano et al. (2017), [“Estimating Model-Error Covariances in Nonlinear
# State-Space Models using Kalman Smoothing and the Expectation–Maximization
# Algorithm”](https://doi.org/10.1002/qj.3048), *Quarterly Journal of the Royal
# Meteorological Society*, 143(705), 1877–1885.
#
# The upstream notebook uses a compiled Fortran model for speed. Here, one
# classical fourth-order Runge-Kutta step implements each discrete-time
# transition. Set `CEDA_QUICK_DEMO=true` to run a short version for smoke tests;
# the default settings match the notebook.
#
# This is the scope of the upstream notebook: it demonstrates EM-EKS and
# EM-EnKS on Lorenz-63, then EM-EnKS on Lorenz-96. It does not demonstrate the
# separate ETKS or adaptive/experimental algorithms found elsewhere in CEDA.

using AnalogDataAssimilation
using LinearAlgebra
using Plots
using Random
using Statistics

const QUICK_DEMO = get(ENV, "CEDA_QUICK_DEMO", "false") == "true"

function rk4_step(f, x, dt)
    k1 = f(x)
    k2 = f(x .+ (dt / 2) .* k1)
    k3 = f(x .+ (dt / 2) .* k2)
    k4 = f(x .+ dt .* k3)
    return x .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function generate_truth(dynamics, initial_state, steps, Q, rng)
    nx = length(initial_state)
    states = zeros(Float64, nx, steps + 1)
    states[:, 1] .= initial_state
    noise_factor = cholesky(Symmetric(Q)).L
    for t in 1:steps
        states[:, t + 1] .= dynamics(@view(states[:, t])) + noise_factor * randn(rng, nx)
    end
    return states
end

function generate_observations(observation_operator, states, R, interval, rng)
    no, steps = size(R, 1), size(states, 2) - 1
    observations = fill(NaN, no, steps)
    noise_factor = cholesky(Symmetric(R)).L
    for t in 1:interval:steps
        observations[:, t] .= observation_operator(@view(states[:, t + 1])) +
                              noise_factor * randn(rng, no)
    end
    return observations
end

function climatology(states)
    background = vec(mean(states, dims = 2))
    centered = states .- background
    covariance = centered * centered' ./ (size(states, 2) - 1)
    return background, covariance
end

function compare_em_results(eks, enks, Qtrue, Rtrue, nx, no, title)
    iterations = length(eks.loglikelihood)
    em_steps = 0:iterations

    pQ = plot(
        em_steps,
        [tr(eks.model_noise_covariance[:, :, i]) / nx for i in 1:iterations+1];
        label = "EM-EKS",
        color = :blue,
        xlabel = "EM iteration",
        ylabel = "tr(Q) / n",
        title = "Model-error covariance",
    )
    plot!(pQ, em_steps, [tr(enks.model_noise_covariance[:, :, i]) / nx for i in 1:iterations+1];
          label = "EM-EnKS", color = :red)
    hline!(pQ, [tr(Qtrue) / nx]; label = "true Q", color = :black, linestyle = :dash)

    pR = plot(
        em_steps,
        [tr(eks.observation_noise_covariance[:, :, i]) / no for i in 1:iterations+1];
        label = "EM-EKS",
        color = :blue,
        xlabel = "EM iteration",
        ylabel = "tr(R) / n",
        title = "Observation-error covariance",
    )
    plot!(pR, em_steps, [tr(enks.observation_noise_covariance[:, :, i]) / no for i in 1:iterations+1];
          label = "EM-EnKS", color = :red)
    hline!(pR, [tr(Rtrue) / no]; label = "true R", color = :black, linestyle = :dash)

    pL = plot(1:iterations, eks.loglikelihood; label = "EM-EKS", color = :blue,
              xlabel = "EM iteration", ylabel = "Log-likelihood", title = "Likelihood")
    plot!(pL, 1:iterations, enks.loglikelihood; label = "EM-EnKS", color = :red)

    pE = plot(1:iterations, eks.rmse; label = "EM-EKS", color = :blue,
              xlabel = "EM iteration", ylabel = "RMSE", title = "Smoothed-state error")
    plot!(pE, 1:iterations, enks.rmse; label = "EM-EnKS", color = :red)

    return plot(pQ, pR, pL, pE; layout = (2, 2), size = (1000, 700), plot_title = title)
end

# ## Lorenz-63: comparing EM-EKS and EM-EnKS
#
# The first experiment uses the Lorenz-63 system with ``dt=0.01``,
# ``Q=0.05I``, and ``R=2I``. Observations are available every five model steps.
# The original notebook uses a 1,000-step sequence, 500 EM iterations, and 100
# ensemble members.

Random.seed!(9)
rng_l63 = MersenneTwister(9)
nx_l63 = 3
dt_l63 = 0.01
sigma, rho, beta = 10.0, 28.0, 8 / 3

function lorenz63_rhs(x)
    return [
        sigma * (x[2] - x[1]),
        x[1] * (rho - x[3]) - x[2],
        x[1] * x[2] - beta * x[3],
    ]
end

function lorenz63_rhs_jacobian(x)
    return [
        -sigma sigma 0.0
        rho - x[3] -1.0 -x[1]
        x[2] x[1] -beta
    ]
end

function lorenz63_transition(x)
    return rk4_step(lorenz63_rhs, x, dt_l63)
end

function lorenz63_jacobian(x)
    identity_jacobian = Matrix{Float64}(I, nx_l63, nx_l63)
    k1 = lorenz63_rhs(x)
    J1 = lorenz63_rhs_jacobian(x)

    x2 = x .+ (dt_l63 / 2) .* k1
    J2 = lorenz63_rhs_jacobian(x2) * (identity_jacobian .+ (dt_l63 / 2) .* J1)
    k2 = lorenz63_rhs(x2)

    x3 = x .+ (dt_l63 / 2) .* k2
    J3 = lorenz63_rhs_jacobian(x3) * (identity_jacobian .+ (dt_l63 / 2) .* J2)
    k3 = lorenz63_rhs(x3)

    x4 = x .+ dt_l63 .* k3
    J4 = lorenz63_rhs_jacobian(x4) * (identity_jacobian .+ dt_l63 .* J3)
    return identity_jacobian .+ (dt_l63 / 6) .* (J1 .+ 2 .* J2 .+ 2 .* J3 .+ J4)
end

h_l63(x) = copy(x)
H_l63 = Matrix{Float64}(I, nx_l63, nx_l63)
steps_l63 = QUICK_DEMO ? 100 : 1000
iterations_l63 = QUICK_DEMO ? 5 : 500
members_l63 = QUICK_DEMO ? 32 : 100
Qtrue_l63 = 0.05 .* Matrix{Float64}(I, nx_l63, nx_l63)
Rtrue_l63 = 2.0 .* Matrix{Float64}(I, nx_l63, nx_l63)
initial_l63 = [6.39435776, 9.23172442, 19.15323224]
truth_l63 = generate_truth(lorenz63_transition, initial_l63, steps_l63, Qtrue_l63, rng_l63)
observations_l63 = generate_observations(h_l63, truth_l63, Rtrue_l63, 5, rng_l63)
time_l63 = (1:steps_l63) .* dt_l63

# The background mean and covariance are estimated from the simulated truth,
# as in CEDA. The initial ``Q`` and ``R`` guesses are both identity matrices.

xb_l63, B_l63 = climatology(truth_l63)
Qinit_l63 = Matrix{Float64}(I, nx_l63, nx_l63)
Rinit_l63 = Matrix{Float64}(I, nx_l63, nx_l63)

# Plot true state trajectories and the intermittent observations.

p_l63_data = plot(time_l63, truth_l63[:, 2:end]'; label = ["x₁" "x₂" "x₃"],
                  xlabel = "Lorenz-63 time", ylabel = "State",
                  title = "Lorenz-63: truth and observations")
for variable in 1:nx_l63
    scatter!(p_l63_data, time_l63, observations_l63[variable, :];
             label = "observed x$(variable)", markersize = 2)
end
p_l63_data

# Fit both smoothers using the same model, observations, and initial
# distribution. The EKS uses the exact Jacobian of the RK4 transition and the
# identity observation Jacobian.

eks_l63 = em_eks(
    observations_l63;
    initial_state = xb_l63,
    initial_covariance = B_l63,
    model_noise_covariance = Qinit_l63,
    observation_noise_covariance = Rinit_l63,
    dynamics = lorenz63_transition,
    dynamics_jacobian = lorenz63_jacobian,
    observation_operator = h_l63,
    observation_jacobian = _ -> H_l63,
    iterations = iterations_l63,
    truth = truth_l63,
)

enks_l63 = em_enks(
    observations_l63;
    initial_state = xb_l63,
    initial_covariance = B_l63,
    model_noise_covariance = Qinit_l63,
    observation_noise_covariance = Rinit_l63,
    dynamics = lorenz63_transition,
    observation_matrix = H_l63,
    ensemble_size = members_l63,
    iterations = iterations_l63,
    truth = truth_l63,
    rng = rng_l63,
)

# Compare normalized traces of ``Q`` and ``R`` to their true values, plus
# observed-data likelihood and smoothed-state RMSE.

p_l63_comparison = compare_em_results(
    eks_l63, enks_l63, Qtrue_l63, Rtrue_l63, nx_l63, nx_l63,
    "Lorenz-63: EM-EKS and EM-EnKS",
)

# CEDA reports that EM-EnKS approaches the true covariance values more
# closely, while EM-EKS performs poorly with the relatively sparse
# observations. Increasing the ensemble size can improve the EnKS estimates.

# ## Lorenz-96: EM-EnKS in a higher-dimensional system
#
# The second experiment uses Lorenz-96 with 40 state variables, forcing
# ``F=8``, and ``dt=0.05``. The original notebook generates a 450-step
# trajectory, discards its first 50 states as spin-up, and observes all
# variables every two steps. The EM-EnKS uses 100 members and 50 iterations.

rng_l96 = MersenneTwister(1)
nx_l96 = 40
dt_l96 = 0.05
forcing_l96 = 8.0

function lorenz96_rhs(x)
    return [
        (x[mod1(j + 1, nx_l96)] - x[mod1(j - 2, nx_l96)]) *
        x[mod1(j - 1, nx_l96)] - x[j] + forcing_l96
        for j in 1:nx_l96
    ]
end

lorenz96_transition(x) = rk4_step(lorenz96_rhs, x, dt_l96)

steps_l96 = QUICK_DEMO ? 70 : 450
spinup_l96 = QUICK_DEMO ? 10 : 50
fit_steps_l96 = QUICK_DEMO ? 60 : 400
iterations_l96 = QUICK_DEMO ? 2 : 50
members_l96 = QUICK_DEMO ? 40 : 100
Qtrue_l96 = 0.1 .* Matrix{Float64}(I, nx_l96, nx_l96)
Rtrue_l96 = 2.0 .* Matrix{Float64}(I, nx_l96, nx_l96)
truth_l96_all = generate_truth(
    lorenz96_transition, zeros(nx_l96), steps_l96, Qtrue_l96, rng_l96,
)
observations_l96_all = generate_observations(
    identity, truth_l96_all, Rtrue_l96, 2, rng_l96,
)

# Retain the same 400 forecast transitions after the 50-step spin-up used in
# the notebook. The initial state and its successor are both included in
# `truth_l96`, while its 400 observation columns correspond to the transitions.

truth_l96 = truth_l96_all[:, spinup_l96+1:spinup_l96+fit_steps_l96+1]
observations_l96 = observations_l96_all[:, spinup_l96+1:spinup_l96+fit_steps_l96]
time_l96 = (0:fit_steps_l96) .* dt_l96
xb_l96, B_l96 = climatology(truth_l96)
Qinit_l96 = Matrix{Float64}(I, nx_l96, nx_l96)
Rinit_l96 = Matrix{Float64}(I, nx_l96, nx_l96)
H_l96 = Matrix{Float64}(I, nx_l96, nx_l96)

# The panels show all 40 truth and observed trajectories; missing observations
# remain blank in the second heatmap.

p_l96_truth = heatmap(
    time_l96, 1:nx_l96, truth_l96;
    color = :jet, clims = (-10, 10), xlabel = "Lorenz-96 time",
    ylabel = "Variable", title = "True trajectories",
)
p_l96_observations = heatmap(
    time_l96[2:end], 1:nx_l96, observations_l96;
    color = :jet, clims = (-10, 10), xlabel = "Lorenz-96 time",
    ylabel = "Variable", title = "Observed trajectories",
)
p_l96_data = plot(p_l96_truth, p_l96_observations; layout = (1, 2), size = (1000, 450))

# Estimate ``Q`` and ``R`` with EM-EnKS. Since the Lorenz-96 observation
# operator is identity, the notebook's ensemble-smoother experiment transfers
# directly to `em_enks`.

enks_l96 = em_enks(
    observations_l96;
    initial_state = xb_l96,
    initial_covariance = B_l96,
    model_noise_covariance = Qinit_l96,
    observation_noise_covariance = Rinit_l96,
    dynamics = lorenz96_transition,
    observation_matrix = H_l96,
    ensemble_size = members_l96,
    iterations = iterations_l96,
    truth = truth_l96,
    rng = rng_l96,
)

# Plot the normalized covariance traces, log-likelihood, and state RMSE against
# the known values and the initial parameter guesses.

iterations_l96_axis = 0:iterations_l96
p_l96_Q = plot(
    iterations_l96_axis,
    [tr(enks_l96.model_noise_covariance[:, :, i]) / nx_l96 for i in 1:iterations_l96+1];
    label = "EM-EnKS", color = :red, xlabel = "EM iteration",
    ylabel = "tr(Q) / n", title = "Model-error covariance",
)
hline!(p_l96_Q, [tr(Qtrue_l96) / nx_l96]; label = "true Q", color = :black, linestyle = :dash)
plot!(p_l96_Q, [0, iterations_l96], [1, 1]; label = "initial Q", color = :gray, linestyle = :dot)

p_l96_R = plot(
    iterations_l96_axis,
    [tr(enks_l96.observation_noise_covariance[:, :, i]) / nx_l96 for i in 1:iterations_l96+1];
    label = "EM-EnKS", color = :red, xlabel = "EM iteration",
    ylabel = "tr(R) / n", title = "Observation-error covariance",
)
hline!(p_l96_R, [tr(Rtrue_l96) / nx_l96]; label = "true R", color = :black, linestyle = :dash)
plot!(p_l96_R, [0, iterations_l96], [1, 1]; label = "initial R", color = :gray, linestyle = :dot)

p_l96_likelihood = plot(1:iterations_l96, enks_l96.loglikelihood; label = "EM-EnKS",
                        color = :red, xlabel = "EM iteration",
                        ylabel = "Log-likelihood", title = "Likelihood")
p_l96_rmse = plot(1:iterations_l96, enks_l96.rmse; label = "EM-EnKS",
                  color = :red, xlabel = "EM iteration", ylabel = "RMSE",
                  title = "Smoothed-state error")
p_l96_results = plot(
    p_l96_Q, p_l96_R, p_l96_likelihood, p_l96_rmse;
    layout = (2, 2), size = (1000, 700), plot_title = "Lorenz-96: EM-EnKS",
)

# In the upstream experiment, the 100-member EnKS estimates approach the
# true ``Q`` and ``R``. A longer sequence (for example 1,000 steps) can improve
# the estimates further. The optional quick mode is intended only to verify
# this example's code path; use the default settings for the published setup.
