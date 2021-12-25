import DifferentialEquations: ODEProblem, EnsembleProblem, solve, remake, Tsit5

""" 

    Apply the dynamical models to generate numerical forecasts. 

"""
function (ssm::StateSpaceModel)(x::Array{Float64,2})

    nv, np = size(x)
	xf = similar(x)
    p = ssm.params
    tspan = (0.0, ssm.dt_integration)
    u0 = zeros(Float64, nv)

    prob = ODEProblem(ssm.model, u0, tspan, p)

    function prob_func(prob, i, repeat)
		remake(prob,u0 = vec(x[:, i]))
    end

    monte_prob = EnsembleProblem(prob, prob_func = prob_func)

	sim = solve(monte_prob, Tsit5(); trajectories = np, save_everystep = false)

	for i = 1:np
		xf[:, i] .= last(sim[i].u)
    end

	xf, xf

end
