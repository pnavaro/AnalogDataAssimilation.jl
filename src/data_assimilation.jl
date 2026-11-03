using Distributions
using LinearAlgebra
using ProgressMeter

abstract type MonteCarloMethod end

export DataAssimilation

"""
    DataAssimilation( forecasting, method, np, xt, sigma2) 

parameters of the filtering method
 - method :chosen method (:AnEnKF, :AnEnKS, :AnPF)
 - N      : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
struct DataAssimilation

    xb::Vector{Float64}
    B::Array{Float64,2}
    H::Array{Bool,2}
    R::Array{Float64,2}
    m::AbstractForecasting

    function DataAssimilation(m::AbstractForecasting, xt::TimeSeries, sigma2::Float64)

        xb = xt.u[1]
        B = 0.1 * Matrix(I, xt.nv, xt.nv)
        H = Matrix(I, xt.nv, xt.nv)
        R = sigma2 .* H

        new(xb, B, H, R, m)

    end

    function DataAssimilation(m::StateSpaceModel, xt::TimeSeries)

        xb = xt.u[1]
        B = 0.1 * Matrix(I, xt.nv, xt.nv)
        H = Matrix(I, xt.nv, xt.nv)
        R = m.sigma2_obs .* H

        new(xb, B, H, R, m)

    end

end

function _validate_assimilation_inputs(da::DataAssimilation, yo::TimeSeries, np::Integer)
    np >= 2 || throw(ArgumentError("ensemble size must be at least 2"))
    yo.nt > 0 || throw(ArgumentError("observations must contain at least one time step"))
    size(da.H, 2) == length(da.xb) ||
        throw(DimensionMismatch("observation matrix state dimension must match the background state"))
    size(da.H, 1) == yo.nv ||
        throw(DimensionMismatch("observation variable count must match the observation matrix"))
    size(da.R) == (yo.nv, yo.nv) ||
        throw(DimensionMismatch("observation covariance must match the observation dimension"))
    size(da.B) == (length(da.xb), length(da.xb)) ||
        throw(DimensionMismatch("background covariance must match the state dimension"))
    length(yo.u) == yo.nt ||
        throw(DimensionMismatch("observation time series length must match nt"))
    for (time, observation) in enumerate(yo.u)
        length(observation) == yo.nv ||
            throw(DimensionMismatch("observation at time $time must have length $(yo.nv)"))
        any(isinf, observation) &&
            throw(ArgumentError("observations may contain NaN for missing values, but not Inf"))
    end
    return nothing
end

_forecast_with_rng(model::AbstractForecasting, ensemble, rng) = model(ensemble)
