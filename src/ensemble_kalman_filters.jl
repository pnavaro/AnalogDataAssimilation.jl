using Distributions
using LinearAlgebra
using Random
using Statistics

export AnEnKF

"""
    AnEnKF(np)

Construct an ensemble Kalman filter with at least two members. `forecast`
accepts an `rng` keyword (defaulting to Julia's task-local RNG) for reproducible
sampling. Missing observations are represented by `NaN`; infinite observations
are rejected.
"""
struct AnEnKF <: MonteCarloMethod

    np::Int64

    function AnEnKF(np::Integer)
        np >= 2 || throw(ArgumentError("AnEnKF requires at least 2 ensemble members"))
        new(Int64(np))
    end

end


""" 
    forecast(da, yo, mc; progress=true, rng=Random.default_rng())

- da: DataAssimilation
- yo: Observations
- mc: Monte Carlo method
- progress: display the progress bar.

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function forecast(da::DataAssimilation, yo::TimeSeries, mc::AnEnKF; progress = true, rng = Random.default_rng())

    # dimensions
    nt = yo.nt        # number of observations
    np = mc.np        # number of particles
    nv = yo.nv        # number of variables (dimensions of problem)
    _validate_assimilation_inputs(da, yo, np)

    # initialization
    x̂ = TimeSeries(nt, nv)

    part = [zeros(Float64, (nv, np)) for i = 1:nt]
    xf = zeros(Float64, (nv, np))
    xf_mean = similar(xf)

    if progress
        p = Progress(nt)
    end

    for k = 1:nt

        if progress
            next!(p)
        end

        # update step (compute forecasts)            
        if k == 1
            xf .= rand(rng, MvNormal(da.xb, da.B), np)
        else
            xf, xf_mean = _forecast_with_rng(da.m, part[k-1], rng)
        end

        part[k] .= xf

        forecast_anomalies = xf .- mean(xf, dims = 2)
        forecast_covariance = forecast_anomalies * forecast_anomalies' ./ (np - 1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

        if n > 0
            μ = zeros(Float64, n)
            σ = da.R[ivar_obs, ivar_obs]
            eps = rand(rng, MvNormal(μ, σ), np)
            yf = da.H[ivar_obs, :] * xf
            Σ = Symmetric(
                (da.H[ivar_obs, :] * forecast_covariance) * da.H[ivar_obs, :]' +
                da.R[ivar_obs, ivar_obs],
            )
            K = (forecast_covariance * da.H[ivar_obs, :]') / Σ
            d = yo.u[k][ivar_obs] .- yf .+ eps
            part[k] .= xf .+ K * d
            # compute likelihood
            innov_ll = mean(yo.u[k][ivar_obs] .- yf, dims = 2)
        else
            part[k] .= xf
        end

        x̂.u[k] .= vec(sum(part[k] ./ np, dims = 2))

    end

    x̂

end
