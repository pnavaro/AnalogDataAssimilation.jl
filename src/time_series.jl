abstract type AbstractTimeSeries end

export TimeSeries

struct TimeSeries <: AbstractTimeSeries

    ntime::Integer
    nvalues::Integer
    time::Vector{Float64}
    values::Vector{Array{Float64,1}}

    function TimeSeries(ntime::Integer, nvalues::Integer)

        time = zeros(Float64, ntime)
        values = [zeros(Float64, nvalues) for i = 1:ntime]

        new(ntime, nvalues, time, values)

    end

    function TimeSeries(time::Array{Float64,1}, values::Array{Array{Float64,1}})

        ntime = length(time)
        nvalues = size(first(values))[1]

        new(ntime, nvalues, time, values)

    end

end

import Base: getindex

function getindex(x::TimeSeries, i::Int)
    getindex.(x.values, i)
end

import Base: -

function (-)(a::TimeSeries, b::TimeSeries)

    @assert a.nvalues == b.nvalues
    @assert all(a.time .== b.time)

    TimeSeries(a.time, [a[i] - b[i] for i = 1:a.ntime])

end

"""
    RMSE(a, b)

Compute the Root Mean Square Error between 2 time series.
"""
function RMSE(a, b)

    sqrt(mean((vcat(a.values'...) .- vcat(b.values'...)) .^ 2))

end

"""
    train_test_split( X, Y; test_size)

Split time series into random train and test subsets
"""
function train_test_split(X::TimeSeries, Y::TimeSeries; test_size = 0.5)

    time = X.time
    T = length(time)
    T_test = Int64(T * test_size)
    T_train = T - T_test

    X_train = TimeSeries(time[1:T_train], X.values[:, 1:T_train-1])
    Y_train = TimeSeries(time[2:end], Y.values[:, 1:T_train-1])

    X_test = TimeSeries(time, X.values[:, T_train:end])
    Y_test = TimeSeries(time[2:end], Y.values[:, T_train:end-1])

    X_train, Y_train, X_test, Y_test

end
