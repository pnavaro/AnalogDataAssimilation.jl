using LinearAlgebra
using Random
using Statistics

export CovarianceEMResult, em_eks, em_enks

"""
    CovarianceEMResult

Results from an EM covariance-estimation run. State arrays use the layout
`variables × time × EM iteration`; smoothed covariance arrays use
`row × column × time × EM iteration`. Parameter covariance histories use
`row × column × EM iteration`, including the initial parameters at index 1.
"""
struct CovarianceEMResult
    algorithm::Symbol
    smoothed_states::Array{Float64,3}
    smoothed_covariances::Array{Float64,4}
    background_state::Matrix{Float64}
    background_covariance::Array{Float64,3}
    model_noise_covariance::Array{Float64,3}
    observation_noise_covariance::Array{Float64,3}
    loglikelihood::Vector{Float64}
    rmse::Union{Nothing,Vector{Float64}}
end

function _em_observations(observations)
    Y = Matrix{Float64}(observations)
    size(Y, 1) > 0 || throw(ArgumentError("observations must have at least one variable"))
    size(Y, 2) > 0 || throw(ArgumentError("observations must contain at least one time step"))
    any(isinf, Y) && throw(ArgumentError("observations may contain NaN for missing values, but not Inf"))
    return Y
end

function _em_covariance(name, covariance, n)
    C = Matrix{Float64}(covariance)
    size(C) == (n, n) || throw(DimensionMismatch("$name must have size ($n, $n)"))
    all(isfinite, C) || throw(ArgumentError("$name must contain only finite values"))
    isapprox(C, C'; rtol = 1e-8, atol = 1e-10) ||
        throw(ArgumentError("$name must be symmetric"))
    C .= (C .+ C') ./ 2
    minimum(eigvals(Symmetric(C))) >= -1e-9 ||
        throw(ArgumentError("$name must be positive semidefinite"))
    return C
end

function _em_psd(C)
    E = eigen(Symmetric((C .+ C') ./ 2))
    return E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors'
end

function _em_covariance_factor(C)
    E = eigen(Symmetric(C))
    return E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
end

function _em_prepare(
    observations,
    initial_state,
    initial_covariance,
    model_noise_covariance,
    observation_noise_covariance;
    iterations = iterations,
    model_noise_structure = model_noise_structure,
    model_noise_template = model_noise_template,
    truth = truth,
)
    xb = Vector{Float64}(initial_state)
    nx = length(xb)
    nx > 0 || throw(ArgumentError("initial_state must not be empty"))
    all(isfinite, xb) || throw(ArgumentError("initial_state must contain only finite values"))

    Y = _em_observations(observations)
    B = _em_covariance("initial_covariance", initial_covariance, nx)
    Q = _em_covariance("model_noise_covariance", model_noise_covariance, nx)
    R = _em_covariance("observation_noise_covariance", observation_noise_covariance, size(Y, 1))

    iterations isa Integer && iterations > 0 ||
        throw(ArgumentError("iterations must be a positive integer"))
    structure = model_noise_structure === :diagonal ? :diag :
                model_noise_structure === :scaled ? :const : model_noise_structure
    structure in (:full, :diag, :const) ||
        throw(ArgumentError("model_noise_structure must be :full, :diag, or :const"))

    template = nothing
    if structure === :const
        model_noise_template === nothing &&
            throw(ArgumentError("model_noise_template is required for :const structure"))
        template = _em_covariance("model_noise_template", model_noise_template, nx)
        isposdef(Symmetric(template)) ||
            throw(ArgumentError("model_noise_template must be positive definite"))
    end

    truth_matrix = nothing
    if truth !== nothing
        truth_matrix = Matrix{Float64}(truth)
        size(truth_matrix) == (nx, size(Y, 2) + 1) ||
            throw(DimensionMismatch("truth must have size ($nx, $(size(Y, 2) + 1))"))
        all(isfinite, truth_matrix) || throw(ArgumentError("truth must contain only finite values"))
    end

    return (Y, xb, B, Q, R, Int(iterations), structure, template, truth_matrix)
end

function _em_vector_result(value, n, name)
    result = Vector{Float64}(value)
    length(result) == n || throw(DimensionMismatch("$name returned $(length(result)) values; expected $n"))
    all(isfinite, result) || throw(ArgumentError("$name returned non-finite values"))
    return result
end

function _em_matrix_result(value, rows, columns, name)
    result = Matrix{Float64}(value)
    size(result) == (rows, columns) ||
        throw(DimensionMismatch("$name returned size $(size(result)); expected ($rows, $columns)"))
    all(isfinite, result) || throw(ArgumentError("$name returned non-finite values"))
    return result
end

function _em_smoother(
    Y,
    xb,
    B,
    Q,
    R,
    dynamics,
    dynamics_jacobian,
    observation_operator,
    observation_jacobian,
    inflation,
)
    nx, nt = length(xb), size(Y, 2)
    no = size(Y, 1)
    Xa = zeros(Float64, nx, nt + 1)
    Pa = zeros(Float64, nx, nx, nt + 1)
    Xf = zeros(Float64, nx, nt)
    Pf = zeros(Float64, nx, nx, nt)
    F_all = zeros(Float64, nx, nx, nt)
    Xa[:, 1] .= xb
    Pa[:, :, 1] .= B

    loglikelihood = 0.0
    for t in 1:nt
        x_previous = @view Xa[:, t]
        F = _em_matrix_result(dynamics_jacobian(copy(x_previous)), nx, nx, "dynamics_jacobian")
        x_forecast = _em_vector_result(dynamics(copy(x_previous)), nx, "dynamics")
        P_forecast = _em_psd(F * Pa[:, :, t] * F' + Q)
        F_all[:, :, t] .= F
        Xf[:, t] .= x_forecast
        Pf[:, :, t] .= P_forecast

        observed = findall(isfinite, @view Y[:, t])
        if isempty(observed)
            Xa[:, t + 1] .= x_forecast
            Pa[:, :, t + 1] .= P_forecast
            continue
        end

        predicted_observation = _em_vector_result(
            observation_operator(x_forecast), no, "observation_operator",
        )
        H = _em_matrix_result(
            observation_jacobian(x_forecast), no, nx, "observation_jacobian",
        )
        H_observed = H[observed, :]
        R_observed = R[observed, observed]
        innovation = Y[observed, t] - predicted_observation[observed]
        S_update = Symmetric(H_observed * P_forecast * H_observed' + R_observed ./ inflation)
        cholesky(S_update)
        K = (P_forecast * H_observed') / S_update
        x_analysis = x_forecast + K * innovation
        residual_map = Matrix{Float64}(I, nx, nx) - K * H_observed
        P_analysis = _em_psd(
            residual_map * P_forecast * residual_map' + K * (R_observed ./ inflation) * K',
        )

        Xa[:, t + 1] .= x_analysis
        Pa[:, :, t + 1] .= P_analysis
        S_likelihood = Symmetric(H_observed * P_forecast * H_observed' + R_observed)
        likelihood_factor = cholesky(S_likelihood)
        loglikelihood += -0.5 * (
            length(observed) * log(2π) +
            2 * sum(log, diag(likelihood_factor.L)) +
            dot(innovation, likelihood_factor \ innovation)
        )
    end

    Xs = copy(Xa)
    Ps = copy(Pa)
    smoother_gains = zeros(Float64, nx, nx, nt)
    for t in nt:-1:1
        J = Pa[:, :, t] * F_all[:, :, t]' * pinv(Pf[:, :, t])
        smoother_gains[:, :, t] .= J
        Xs[:, t] .= Xa[:, t] + J * (Xs[:, t + 1] - Xf[:, t])
        Ps[:, :, t] .= _em_psd(
            Pa[:, :, t] + J * (Ps[:, :, t + 1] - Pf[:, :, t]) * J',
        )
    end
    return Xs, Ps, Xa, Pa, Xf, Pf, smoother_gains, loglikelihood
end

function _em_maximize_eks(
    Xs,
    Ps,
    smoother_gains,
    Y,
    R,
    dynamics,
    dynamics_jacobian,
    observation_operator,
    observation_jacobian,
)
    nx, nt = size(Xs, 1), size(Y, 2)
    sumQ = zeros(Float64, nx, nx)
    for t in 1:nt
        x_previous = @view Xs[:, t]
        x_next = @view Xs[:, t + 1]
        F = _em_matrix_result(dynamics_jacobian(copy(x_previous)), nx, nx, "dynamics_jacobian")
        residual = x_next - _em_vector_result(dynamics(copy(x_previous)), nx, "dynamics")
        cross_covariance = Ps[:, :, t + 1] * smoother_gains[:, :, t]'
        sumQ .+= Ps[:, :, t + 1] + residual * residual' +
                 F * Ps[:, :, t] * F' - cross_covariance * F' - F * cross_covariance'
    end
    sumQ .= (sumQ .+ sumQ') ./ 2

    no = size(Y, 1)
    sumR = zeros(Float64, no, no)
    counts = zeros(Int, no, no)
    for t in 1:nt
        observed = findall(isfinite, @view Y[:, t])
        isempty(observed) && continue
        x = @view Xs[:, t + 1]
        predicted = _em_vector_result(observation_operator(copy(x)), no, "observation_operator")
        H = _em_matrix_result(observation_jacobian(copy(x)), no, nx, "observation_jacobian")
        residual = Y[observed, t] - predicted[observed]
        expected_residual = residual * residual' +
                            H[observed, :] * Ps[:, :, t + 1] * H[observed, :]'
        sumR[observed, observed] .+= expected_residual
        counts[observed, observed] .+= 1
    end

    Rnew = copy(R)
    for j in 1:no, i in 1:no
        counts[i, j] > 0 && (Rnew[i, j] = sumR[i, j] / counts[i, j])
    end
    return Xs[:, 1], Ps[:, :, 1], _em_psd(sumQ ./ nt), _em_psd(Rnew)
end

function _em_structured_Q(Qestimate, structure, template)
    if structure === :diag
        return _em_psd(Diagonal(max.(diag(Qestimate), 0.0)) |> Matrix)
    elseif structure === :const
        scale = tr(template \ Qestimate) / size(Qestimate, 1)
        return _em_psd(max(scale, 0.0) .* template)
    end
    return _em_psd(Qestimate)
end

function _em_result(algorithm, smoothed_states, smoothed_covariances, xb_history, B_history, Q_history, R_history, likelihood, truth)
    errors = if truth === nothing
        nothing
    else
        [sqrt(mean(abs2, smoothed_states[:, :, iteration] - truth)) for iteration in axes(smoothed_states, 3)]
    end
    return CovarianceEMResult(
        algorithm,
        smoothed_states,
        smoothed_covariances,
        xb_history,
        B_history,
        Q_history,
        R_history,
        likelihood,
        errors,
    )
end

"""
    em_eks(observations; initial_state, initial_covariance,
           model_noise_covariance, observation_noise_covariance,
           dynamics, dynamics_jacobian, observation_jacobian, ...)

Estimate state and noise covariances with the extended Kalman smoother and EM.
`observations` is an `observation_variables × time` matrix; `NaN` entries mark
missing observations. Dynamics, observation operators, and their Jacobians are
functions of a state vector. Use `model_noise_structure=:full`, `:diag`, or
`:const` (a scalar multiple of `model_noise_template`) to constrain the model
noise estimate.
"""
function em_eks(
    observations::AbstractMatrix;
    initial_state,
    initial_covariance,
    model_noise_covariance,
    observation_noise_covariance,
    dynamics,
    dynamics_jacobian,
    observation_jacobian,
    observation_operator = identity,
    iterations::Integer = 10,
    inflation::Real = 1.0,
    estimate_model_noise::Bool = true,
    estimate_observation_noise::Bool = true,
    estimate_background::Bool = true,
    model_noise_structure::Symbol = :full,
    model_noise_template = nothing,
    truth = nothing,
)
    isfinite(inflation) && inflation > 0 ||
        throw(ArgumentError("inflation must be a positive finite number"))
    Y, xb, B, Q, R, niterations, structure, template, truth_matrix = _em_prepare(
        observations,
        initial_state,
        initial_covariance,
        model_noise_covariance,
        observation_noise_covariance;
        iterations = iterations,
        model_noise_structure = model_noise_structure,
        model_noise_template = model_noise_template,
        truth = truth,
    )
    nx, no = length(xb), size(Y, 1)
    nt = size(Y, 2)
    smoothed_states = zeros(Float64, nx, nt + 1, niterations)
    smoothed_covariances = zeros(Float64, nx, nx, nt + 1, niterations)
    xb_history = zeros(Float64, nx, niterations + 1)
    B_history = zeros(Float64, nx, nx, niterations + 1)
    Q_history = zeros(Float64, nx, nx, niterations + 1)
    R_history = zeros(Float64, no, no, niterations + 1)
    likelihood = zeros(Float64, niterations)
    xb_history[:, 1] .= xb
    B_history[:, :, 1] .= B
    Q_history[:, :, 1] .= Q
    R_history[:, :, 1] .= R

    for iteration in 1:niterations
        Xs, Ps, _, _, _, _, smoother_gains, likelihood[iteration] = _em_smoother(
            Y, xb, B, Q, R, dynamics, dynamics_jacobian, observation_operator,
            observation_jacobian, Float64(inflation),
        )
        smoothed_states[:, :, iteration] .= Xs
        smoothed_covariances[:, :, :, iteration] .= Ps

        xb_new, B_new, Q_new, R_new = _em_maximize_eks(
            Xs, Ps, smoother_gains, Y, R, dynamics, dynamics_jacobian,
            observation_operator, observation_jacobian,
        )
        if estimate_background
            xb, B = xb_new, B_new
        end
        estimate_model_noise && (Q = _em_structured_Q(Q_new, structure, template))
        estimate_observation_noise && (R = R_new)
        xb_history[:, iteration + 1] .= xb
        B_history[:, :, iteration + 1] .= B
        Q_history[:, :, iteration + 1] .= Q
        R_history[:, :, iteration + 1] .= R
    end

    return _em_result(
        :eks, smoothed_states, smoothed_covariances, xb_history, B_history,
        Q_history, R_history, likelihood, truth_matrix,
    )
end

function _em_sample_covariance(ensemble)
    centered = ensemble .- mean(ensemble, dims = 2)
    return centered * centered' ./ (size(ensemble, 2) - 1)
end

function _em_enks_smoother(Y, xb, B, Q, R, dynamics, H, ensemble_size, inflation, rng)
    nx, nt = length(xb), size(Y, 2)
    no = size(Y, 1)
    Xa = zeros(Float64, nx, ensemble_size, nt + 1)
    Xf = zeros(Float64, nx, ensemble_size, nt)
    Xa[:, :, 1] .= xb .+ _em_covariance_factor(B) * randn(rng, nx, ensemble_size)
    Qfactor = _em_covariance_factor(Q)
    likelihood = 0.0

    for t in 1:nt
        x_previous = @view Xa[:, :, t]
        x_forecast = Matrix{Float64}(undef, nx, ensemble_size)
        for member in 1:ensemble_size
            x_forecast[:, member] .= _em_vector_result(
                dynamics(copy(@view x_previous[:, member])), nx, "dynamics",
            )
        end
        x_forecast .+= Qfactor * randn(rng, nx, ensemble_size)
        Xf[:, :, t] .= x_forecast

        observed = findall(isfinite, @view Y[:, t])
        if isempty(observed)
            Xa[:, :, t + 1] .= x_forecast
            continue
        end

        H_observed = H[observed, :]
        R_observed = R[observed, observed]
        predicted_observation = H_observed * x_forecast
        forecast_covariance = _em_sample_covariance(x_forecast)
        S = Symmetric(H_observed * forecast_covariance * H_observed' + R_observed ./ inflation)
        cholesky(S)
        K = (forecast_covariance * H_observed') / S
        observation_perturbation = _em_covariance_factor(R_observed ./ inflation) *
                                   randn(rng, length(observed), ensemble_size)
        innovation = Y[observed, t] .+ observation_perturbation .- predicted_observation
        Xa[:, :, t + 1] .= x_forecast + K * innovation

        mean_innovation = Y[observed, t] - vec(mean(predicted_observation, dims = 2))
        S_likelihood = Symmetric(H_observed * forecast_covariance * H_observed' + R_observed)
        likelihood_factor = cholesky(S_likelihood)
        likelihood += -0.5 * (
            length(observed) * log(2π) +
            2 * sum(log, diag(likelihood_factor.L)) +
            dot(mean_innovation, likelihood_factor \ mean_innovation)
        )
    end

    Xs = copy(Xa)
    for t in nt:-1:1
        forecast = @view Xf[:, :, t]
        analysis = @view Xa[:, :, t]
        cross_covariance = (analysis .- mean(analysis, dims = 2)) *
                           (forecast .- mean(forecast, dims = 2))' ./
                           (ensemble_size - 1)
        forecast_covariance = _em_sample_covariance(forecast)
        smoother_gain = cross_covariance * pinv(forecast_covariance)
        Xs[:, :, t] .= analysis + smoother_gain * (Xs[:, :, t + 1] - forecast)
    end

    return Xs, likelihood
end

function _em_maximize_enks(Xs, Y, H, R, dynamics)
    nx, ensemble_size, nstates = size(Xs)
    nt, no = nstates - 1, size(Y, 1)
    initial_ensemble = @view Xs[:, :, 1]
    xb = vec(mean(initial_ensemble, dims = 2))
    B = _em_psd(_em_sample_covariance(initial_ensemble))

    sumQ = zeros(Float64, nx, nx)
    for t in 1:nt
        residuals = Matrix{Float64}(undef, nx, ensemble_size)
        for member in 1:ensemble_size
            previous = copy(@view Xs[:, member, t])
            residuals[:, member] .= @view(Xs[:, member, t + 1]) .-
                                     _em_vector_result(dynamics(previous), nx, "dynamics")
        end
        sumQ .+= residuals * residuals' ./ ensemble_size
    end

    sumR = zeros(Float64, no, no)
    counts = zeros(Int, no, no)
    for t in 1:nt
        observed = findall(isfinite, @view Y[:, t])
        isempty(observed) && continue
        residuals = Y[observed, t] .- H[observed, :] * Xs[:, :, t + 1]
        sumR[observed, observed] .+= residuals * residuals' ./ ensemble_size
        counts[observed, observed] .+= 1
    end
    Rnew = copy(R)
    for j in 1:no, i in 1:no
        counts[i, j] > 0 && (Rnew[i, j] = sumR[i, j] / counts[i, j])
    end
    return xb, B, _em_psd(sumQ ./ nt), _em_psd(Rnew)
end

"""
    em_enks(observations; initial_state, initial_covariance,
            model_noise_covariance, observation_noise_covariance,
            dynamics, observation_matrix, ...)

Estimate state and noise covariances with the ensemble Kalman smoother and EM.
`observations` is an `observation_variables × time` matrix; `NaN` entries mark
missing observations. The linear observation operator is supplied as
`observation_matrix`. Pass an `AbstractRNG` with `rng` for reproducible runs.
"""
function em_enks(
    observations::AbstractMatrix;
    initial_state,
    initial_covariance,
    model_noise_covariance,
    observation_noise_covariance,
    dynamics,
    observation_matrix,
    ensemble_size::Integer = 100,
    iterations::Integer = 10,
    inflation::Real = 1.0,
    estimate_model_noise::Bool = true,
    estimate_observation_noise::Bool = true,
    estimate_background::Bool = true,
    model_noise_structure::Symbol = :full,
    model_noise_template = nothing,
    truth = nothing,
    rng = Random.default_rng(),
)
    isfinite(inflation) && inflation > 0 ||
        throw(ArgumentError("inflation must be a positive finite number"))
    ensemble_size isa Integer && ensemble_size >= 2 ||
        throw(ArgumentError("ensemble_size must be an integer greater than or equal to 2"))
    Y, xb, B, Q, R, niterations, structure, template, truth_matrix = _em_prepare(
        observations,
        initial_state,
        initial_covariance,
        model_noise_covariance,
        observation_noise_covariance;
        iterations = iterations,
        model_noise_structure = model_noise_structure,
        model_noise_template = model_noise_template,
        truth = truth,
    )
    nx, no = length(xb), size(Y, 1)
    H = _em_matrix_result(observation_matrix, no, nx, "observation_matrix")
    nt = size(Y, 2)
    smoothed_states = zeros(Float64, nx, nt + 1, niterations)
    smoothed_covariances = zeros(Float64, nx, nx, nt + 1, niterations)
    xb_history = zeros(Float64, nx, niterations + 1)
    B_history = zeros(Float64, nx, nx, niterations + 1)
    Q_history = zeros(Float64, nx, nx, niterations + 1)
    R_history = zeros(Float64, no, no, niterations + 1)
    likelihood = zeros(Float64, niterations)
    xb_history[:, 1] .= xb
    B_history[:, :, 1] .= B
    Q_history[:, :, 1] .= Q
    R_history[:, :, 1] .= R

    for iteration in 1:niterations
        smoothed_ensemble, likelihood[iteration] = _em_enks_smoother(
            Y, xb, B, Q, R, dynamics, H, ensemble_size, Float64(inflation), rng,
        )
        smoothed_states[:, :, iteration] .= dropdims(mean(smoothed_ensemble, dims = 2); dims = 2)
        for t in 1:(nt + 1)
            smoothed_covariances[:, :, t, iteration] .=
                _em_sample_covariance(@view smoothed_ensemble[:, :, t])
        end
        xb_new, B_new, Q_new, R_new = _em_maximize_enks(
            smoothed_ensemble, Y, H, R, dynamics,
        )
        if estimate_background
            xb, B = xb_new, B_new
        end
        estimate_model_noise && (Q = _em_structured_Q(Q_new, structure, template))
        estimate_observation_noise && (R = R_new)
        xb_history[:, iteration + 1] .= xb
        B_history[:, :, iteration + 1] .= B
        Q_history[:, :, iteration + 1] .= Q
        R_history[:, :, iteration + 1] .= R
    end

    return _em_result(
        :enks, smoothed_states, smoothed_covariances, xb_history, B_history,
        Q_history, R_history, likelihood, truth_matrix,
    )
end
