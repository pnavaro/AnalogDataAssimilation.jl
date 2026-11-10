@testset " Generate data for Lorenz-63 model " begin

    import OrdinaryDiffEq
    import SciMLBase: ODEProblem, solve

    σ = 10.0
    ρ = 28.0
    β = 8.0 / 3

    dt_integration = 0.01
    dt_states = 1
    dt_obs = 8
    params = [σ, ρ, β]
    var_obs = [1]
    nb_loop_train = 10^2
    nb_loop_test = 10
    sigma2_catalog = 0.0
    sigma2_obs = 2.0


    ssm = StateSpaceModel(
        lorenz63,
        dt_integration,
        dt_states,
        dt_obs,
        params,
        var_obs,
        nb_loop_train,
        nb_loop_test,
        sigma2_catalog,
        sigma2_obs,
    )

    # compute u0 to be in the attractor space
    u0 = [8.0; 0.0; 30.0]
    tspan = (0.0, 5.0)
    prob = ODEProblem(ssm.model, u0, tspan, params)
    u0 = last(solve(prob, reltol = 1e-6, save_everystep = false).u)

    xt, yo, catalog = generate_data(ssm, u0)

    @show last(xt.u)

    @test true

end

@testset "Generated ODE state and catalog continuity" begin
    linear_model(u, p, t) = p .* u
    for nv in (1, 3)
        rates = fill(-0.2, nv)
        ssm = StateSpaceModel(
            linear_model, 0.1, 1, 2, rates, collect(1:nv), 1, 1, 0.0, 0.1,
        )
        initial = collect(1.0:nv)
        sol = solve(ODEProblem(linear_model, initial, (0.0, 0.2), rates))
        u0 = last(sol.u)
        @test u0 isa Vector{Float64}
        @test length(u0) == nv
        xt, yo, catalog = generate_data(ssm, u0)
        @test all(u -> u isa Vector{Float64} && length(u) == nv, xt.u)
        @test all(u -> length(u) == nv, yo.u)
        @test catalog.nv == nv
        @test catalog.data[:, 1] ≈ last(xt.u)
        @test last(xt.u) ≈ u0 .* exp.(rates .* ssm.nb_loop_test) rtol = 1e-4
        @test catalog.data[:, end] ≈
              last(xt.u) .* exp.(rates .* ssm.nb_loop_train) rtol = 1e-4
    end
end
