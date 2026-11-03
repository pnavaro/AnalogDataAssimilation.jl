using Distributions
using LinearAlgebra
using Random
using Statistics

export AnPF

struct AnPF <: MonteCarloMethod
    np::Int64

    function AnPF(np::Integer)
        np >= 2 || throw(ArgumentError("AnPF requires at least 2 particles"))
        new(Int64(np))
    end
end

"""
    forecast(da, yo, AnPF(np); progress=true, rng=Random.default_rng())

Run a sequential importance-resampling particle filter using `np` particles.
`NaN` marks a missing observation; infinite observations are rejected.
Resampled particles receive equal weights. The small Gaussian perturbation
with covariance `da.B / 100` after each forecast is retained from the original
AnDA implementation. Pass an explicit random number generator for
reproducibility.
"""
function forecast(
    da::DataAssimilation,
    yo::TimeSeries,
    mc::AnPF;
    progress = true,
    rng = Random.default_rng(),
)
    nt, np, nv = yo.nt, mc.np, yo.nv
    _validate_assimilation_inputs(da, yo, np)

    estimates = TimeSeries(nt, nv)
    particles = rand(rng, MvNormal(da.xb, da.B), np)
    uniform_weights = fill(1.0 / np, np)
    progress_bar = progress ? Progress(nt) : nothing

    for t in 1:nt
        if progress
            next!(progress_bar)
        end

        if t > 1
            forecast_particles, _ = _forecast_with_rng(da.m, particles, rng)
            size(forecast_particles) == (length(da.xb), np) ||
                throw(DimensionMismatch("forecast must return a state-by-particle matrix"))
            all(isfinite, forecast_particles) ||
                throw(ArgumentError("forecast returned non-finite particles"))
            forecast_particles .+= rand(
                rng,
                MvNormal(zeros(length(da.xb)), da.B ./ 100),
                np,
            )
            particles = forecast_particles
        end

        observed = findall(isfinite, yo.u[t])
        if isempty(observed)
            estimates.u[t] .= particles * uniform_weights
            continue
        end

        H = da.H[observed, :]
        R = Symmetric(da.R[observed, observed])
        observation_noise = MvNormal(zeros(length(observed)), R)
        logweights = Vector{Float64}(undef, np)
        observation = yo.u[t][observed]
        for member in 1:np
            predicted_observation = H * @view(particles[:, member])
            logweights[member] = logpdf(observation_noise, observation - predicted_observation)
        end

        maximum_logweight = maximum(logweights)
        isfinite(maximum_logweight) ||
            throw(ArgumentError("all particle observation likelihoods are zero or non-finite at time $t"))
        weights = exp.(logweights .- maximum_logweight)
        total_weight = sum(weights)
        isfinite(total_weight) && total_weight > 0 ||
            throw(ArgumentError("particle observation weights cannot be normalized at time $t"))
        weights ./= total_weight

        estimates.u[t] .= particles * weights
        ancestors = resample!(Vector{Int64}(undef, np), weights; rng = rng)
        particles = particles[:, ancestors]
    end

    return estimates
end
