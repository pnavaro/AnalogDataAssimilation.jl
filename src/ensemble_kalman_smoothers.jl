using Distributions
using LinearAlgebra
using Random
using Statistics

export AnEnKS

"""
    AnEnKS(np)

Construct an ensemble Kalman smoother with at least two members. `forecast`
accepts an `rng` keyword (defaulting to Julia's task-local RNG) for reproducible
sampling. Missing observations are represented by `NaN`; infinite observations
are rejected.
"""
struct AnEnKS

    np::Int64

    function AnEnKS(np::Integer)
        np >= 2 || throw(ArgumentError("AnEnKS requires at least 2 ensemble members"))
        new(Int64(np))
    end

end

""" 
    forecast(da, yo, AnEnKS(np); rng=Random.default_rng())

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function forecast(da::DataAssimilation, yo::TimeSeries, mc::AnEnKS; progress = true, rng = Random.default_rng())

    # dimensions
    nt = yo.nt        # number of observations
    np = mc.np        # number of particles
    nv = yo.nv        # number of variables (dimensions of problem)
    _validate_assimilation_inputs(da, yo, np)

    # initialization
    x̂ = TimeSeries(nt, nv)

    m_xa_part = [zeros(Float64, (nv, np)) for i = 1:nt]
    xf_part = [zeros(Float64, (nv, np)) for i = 1:nt]
    part = [zeros(Float64, (nv, np)) for i = 1:nt]
    pf = [zeros(Float64, (nv, nv)) for i = 1:nt]
    xf = zeros(Float64, (nv, np))
    xf_mean = similar(xf)
    Ks = zeros(Float64, nv, nv)

    if progress
        p = Progress(nt)
    end

    sampler1 = MvNormal(da.xb, da.B)

    for k = 1:nt
        if progress
            next!(p)
        end

        # update step (compute forecasts)            
        if k == 1
            rand!(rng, sampler1, xf)
        else
            xf, xf_mean = _forecast_with_rng(da.m, part[k-1], rng)
            m_xa_part[k] .= xf_mean
        end

        xf_part[k] .= xf

        forecast_anomalies = xf .- mean(xf, dims = 2)
        pf[k] .= (forecast_anomalies * forecast_anomalies') ./ (np - 1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

        if n > 0

            μ = zeros(Float64, n)
            σ = da.R[ivar_obs, ivar_obs]
            sampler2 = MvNormal(μ, σ)
            eps = rand(rng, sampler2, np)
            yf = da.H[ivar_obs, :] * xf

            Σ = Symmetric(
                (da.H[ivar_obs, :] * pf[k]) * da.H[ivar_obs, :]' +
                da.R[ivar_obs, ivar_obs],
            )
            K = (pf[k] * da.H[ivar_obs, :]') / Σ
            d = yo.u[k][ivar_obs] .- yf .+ eps
            part[k] .= xf .+ K * d

        else

            part[k] .= xf

        end

        x̂.u[k] .= vec(sum(part[k] ./ np, dims = 2))

    end

    if progress
        p = Progress(nt)
    end

    for k = nt:-1:1
        if progress
            next!(p)
        end

        if k == nt
            part[k] .= part[nt]
        else
            tej, m_xa_tmp = _forecast_with_rng(da.m, mean(part[k], dims = 2), rng)
            tmp1 = part[k] .- mean(part[k], dims = 2)
            tmp2 = m_xa_part[k+1] .- m_xa_tmp
            cross_covariance = (tmp1 * tmp2') ./ (np - 1)
            Ks .= cross_covariance * pinv(pf[k+1], rtol = 1e-6)
            part[k] .+= Ks * (part[k+1] .- xf_part[k+1])
        end
        x̂.u[k] .= vec(sum(part[k] ./ np, dims = 2))

    end

    x̂

end
