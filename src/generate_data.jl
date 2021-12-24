using Random, LinearAlgebra

import DifferentialEquations: SDEProblem, solve
import Distributions: MvNormal, rand

export generate_data

"""
    generate_data( ssm, u0; seed=42)

from StateSpace generate:
 - true state (xt)
 - partial/noisy observations (yo)
 - catalog
"""
function generate_data(ssm::StateSpaceModel, u0::Vector{Float64}, seed = 42)

    rng = MersenneTwister(seed)

    try
        @assert ssm.dt_states < ssm.dt_obs
    catch
        @error " ssm.dt_obs must be bigger than ssm.dt_states"
    end

    try
        @assert mod(ssm.dt_obs, ssm.dt_states) == 0.0
    catch
        @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    end

    tspan = (0.0, ssm.nb_loop_test)

    function σ( du, u, p, t)

        for i in eachindex(du)
            du[i] = ssm.sigma2_obs
        end

    end

    prob = SDEProblem(ssm.model, σ, u0, tspan, ssm.params)

    # generSate true state (xt)
    sol = solve(prob, saveat = ssm.dt_states * ssm.dt_integration)
    xt = TimeSeries(sol.t, sol.u)

    # generate  partial/noisy observations (yo)
    nt = xt.nt
    nv = xt.nv

    yo = TimeSeries(xt.t, xt.u .* NaN)
    step = ssm.dt_obs ÷ ssm.dt_states
    nt = length(xt.t)

    d = MvNormal(ssm.sigma2_obs .* Matrix(I, nv, nv))
    ε = rand(d, nt)
    for j = 1:step:nt
        for i in ssm.var_obs
            yo.u[j][i] = xt.u[j][i] + ε[i, j]
        end
    end

    # generate catalog
    u0 = last(sol)
    tspan = (0.0, ssm.nb_loop_train)
    prob = SDEProblem(ssm.model, σ, u0, tspan, ssm.params)
    sol = solve(prob, saveat = ssm.dt_integration)
    n = length(sol.t)

    catalog_tmp = sol.u

    xt, yo, Catalog(hcat(catalog_tmp...), ssm)

end

