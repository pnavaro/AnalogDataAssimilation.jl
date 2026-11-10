@testset " Classic model forecasting " begin

    import OrdinaryDiffEq: Tsit5
    import SciMLBase: ODEProblem, solve

    σ = 10.0
    ρ = 28.0
    β = 8.0 / 3

    dt_integration = 0.01
    dt_states = 1
    dt_obs = 8
    parameters = [σ, ρ, β]
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
        parameters,
        var_obs,
        nb_loop_train,
        nb_loop_test,
        sigma2_catalog,
        sigma2_obs,
    )

    # compute u0 at time = 5 to be in the attractor space

    u0 = [8.0; 0.0; 30.0]
    tspan = (0.0, 5.0)
    prob = ODEProblem(lorenz63, u0, tspan, parameters)
    u0 = last(solve(prob, Tsit5(), reltol = 1e-6, save_everystep = false).u)

    xt, yo, catalog = generate_data(ssm, u0)

    DA = DataAssimilation(ssm, xt)

    x̂ = forecast(DA, yo, AnEnKS(100), progress = false)

    rmse = RMSE(xt, x̂)

    println(" model forecasting RMSE : $rmse ")

    @test rmse < 1.0

end

@testset "ODE ensemble forecast state shape" begin
    linear_model(u, p, t) = p .* u
    for nv in (1, 3)
        rates = -0.2 .* collect(1:nv)
        ssm = StateSpaceModel(
            linear_model, 0.1, 1, 2, rates, collect(1:nv), 1, 1, 0.0, 0.1,
        )
        states = reshape(collect(1.0:3nv), nv, 3)
        expected = states .* exp.(rates .* ssm.dt_integration)
        forecast_states, forecast_mean = ssm(states)
        @test size(forecast_states) == size(states)
        @test forecast_states ≈ expected rtol = 1e-6
        @test forecast_mean ≈ expected rtol = 1e-6
    end
end
