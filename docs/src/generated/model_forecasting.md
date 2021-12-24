```@meta
EditURL = "<unknown>/docs/examples/model_forecasting.jl"
```

# Model Forecasting

## Set the State-Space model

````@example model_forecasting
using Plots
using AnalogDataAssimilation
using DifferentialEquations

σ = 10.0
ρ = 28.0
β = 8.0 / 3

dt_integration = 0.01
dt_states = 1
dt_obs = 8
parameters = [σ, ρ, β]
var_obs = [1]
nb_loop_train = 100
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
````

- compute u0 to be in the attractor space

````@example model_forecasting
u0 = [8.0; 0.0; 30.0]
tspan = (0.0, 5.0)
prob = ODEProblem(ssm.model, u0, tspan, parameters)
u0 = last(solve(prob, reltol = 1e-6, save_everystep = false))
````

## Generate data

````@example model_forecasting
xt, yo, catalog = generate_data(ssm, u0);

plot(xt.t, xt[1])
scatter!(yo.t, yo[1]; markersize = 2)
````

## Data assimilation with model forecasting

````@example model_forecasting
np = 100
DA = DataAssimilation(ssm, xt)
@time x̂ = forecast(DA, yo, PF(np), progress = false);
println(RMSE(xt, x̂))
````

## Plot the times series

````@example model_forecasting
plot(xt.t, x̂[1])
scatter!(xt.t, xt[1]; markersize = 2)
plot!(xt.t, x̂[2])
scatter!(xt.t, xt[2]; markersize = 2)
plot!(xt.t, x̂[3])
scatter!(xt.t, xt[3]; markersize = 2)
````

## Plot the phase-space plot

````@example model_forecasting
p = plot3d(
    1,
    xlim = (-25, 25),
    ylim = (-25, 25),
    zlim = (0, 50),
    title = "Lorenz 63",
    marker = 2,
)
for x in eachrow(vcat(x̂.u'...))
    push!(p, x...)
end
p
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

