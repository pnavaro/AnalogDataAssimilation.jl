abstract type AbstractTimeSeries end

export TimeSeries

struct TimeSeries <: AbstractTimeSeries

    nt::Integer
    nv::Integer
    t::Vector{Float64}
    u::Vector{Array{Float64,1}}

    function TimeSeries(nt::Integer, nv::Integer)

        time = zeros(Float64, nt)
        values = [zeros(Float64, nv) for i = 1:nt]

        new(nt, nv, time, values)

    end

    function TimeSeries(time::Array{Float64,1}, values::Array{Array{Float64,1}})

        nt = length(time)
        nv = size(first(values))[1]

        new(nt, nv, time, values)

    end

end

import Base: getindex

function getindex(x::TimeSeries, i::Int)
    getindex.(x.u, i)
end

import Base: -

function (-)(a::TimeSeries, b::TimeSeries)

    @assert a.nv == b.nv
    @assert all(a.t .== b.t)

    TimeSeries(a.t, [a[i] - b[i] for i = 1:a.nt])

end

"""
    train_test_split( X, Y; test_size)

Split time series into random train and test subsets
"""
function train_test_split(X::TimeSeries, Y::TimeSeries; test_size = 0.5)

    time = X.t
    T = length(time)
    T_test = Int64(T * test_size)
    T_train = T - T_test

    X_train = TimeSeries(time[1:T_train], X.u[:, 1:T_train-1])
    Y_train = TimeSeries(time[2:end], Y.u[:, 1:T_train-1])

    X_test = TimeSeries(time, X.u[:, T_train:end])
    Y_test = TimeSeries(time[2:end], Y.u[:, T_train:end-1])

    X_train, Y_train, X_test, Y_test

end

"""
    RMSE(a, b)

Compute the Root Mean Square Error between 2 time series.
"""
function RMSE(a, b)

    sqrt(mean((vcat(a.u'...) .- vcat(b.u'...)) .^ 2))

end
