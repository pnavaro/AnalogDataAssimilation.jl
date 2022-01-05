var documenterSearchIndex = {"docs":
[{"location":"data_assimilation/#Data-assimilation","page":"Data Assimilation","title":"Data assimilation","text":"","category":"section"},{"location":"data_assimilation/","page":"Data Assimilation","title":"Data Assimilation","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"data_assimilation.jl\"]\nOrder   = [:type, :function]","category":"page"},{"location":"data_assimilation/#AnalogDataAssimilation.DataAssimilation","page":"Data Assimilation","title":"AnalogDataAssimilation.DataAssimilation","text":"DataAssimilation( forecasting, method, np, xt, sigma2)\n\nparameters of the filtering method\n\nmethod :chosen method (:AnEnKF, :AnEnKS, :AnPF)\nN      : number of members (AnEnKF/AnEnKS) or particles (AnPF)\n\n\n\n\n\n","category":"type"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"EditURL = \"https://github.com/pnavaro/AnalogDataAssimilation.jl/blob/master/docs/examples/lorenz63.jl\"","category":"page"},{"location":"generated/lorenz63/#Lorenz-63","page":"Lorenz 63","title":"Lorenz 63","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"Data assimilation are numerical methods used in geosciences to mix the information of observations (noted as y) and a dynamical model (noted as f) in order to estimate the true/hidden state of the system (noted as x) at every time step k. Usually, they are related following a nonlinear state-space model:","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"beginaligned\nx_k  = f(x_k-1) + eta_k \ny_k  = H x_k + epsilon_k\nendaligned","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"with eta and epsilon some independant white Gaussian noises respectively respresenting the model forecast error and the error of observation.","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"In classical data assimilation, we require multiple runs of an explicit dynamical model f with possible severe limitations including the computational cost, the lack of consistency of the model with respect to the observed data as well as modeling uncertainties. Here, an alternative strategy is explored by developing a fully data-driven assimilation. No explicit knowledge of the dynamical model is required. Only a representative catalog of trajectories of the system is assumed to be available. Based on this catalog, the Analog Data Assimilation (AnDA) is introduced by combining machine learning with the analog method (or nearest neighbor search) and stochastic assimilation techniques including Ensemble Kalman Filter and Smoother (AnEnKF, AnEnKS) and Particle Filter (AnPF). We test the accuracy of the technic on different chaotic dynamical models, the Lorenz-63 and Lorenz-96 systems.  # # This Julia program is derived from the Python library is attached to the following publication: # Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. Monthly Weather Review, 145(10), 4093-4107.  # If you use this library, please do not forget to cite this work.","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"using Plots, DifferentialEquations, AnalogDataAssimilation","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"To begin, as dynamical model f, we use the Lorenz-63 chaotic system. First, we generate simulated trajectories from this dynamical model and store them into the catalog. Then, we use this catalog to emulate the dynamical model and we apply the analog data assimilation. Finally, we compare the results of this data-driven approach to the classical data assimilation (using the true Lorenz-63 equations as dynamical model.","category":"page"},{"location":"generated/lorenz63/#Generate-simulated-data","page":"Lorenz 63","title":"Generate simulated data","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"σ = 10.0\nρ = 28.0\nβ = 8.0 / 3\n\ndt_integration = 0.01\ndt_states = 1\ndt_obs = 8\nparameters = [σ, ρ, β]\nvar_obs = [1]\nnb_loop_train = 100\nnb_loop_test = 10\nsigma2_catalog = 0.0\nsigma2_obs = 2.0\n\nssm = StateSpaceModel(\n    lorenz63,\n    dt_integration,\n    dt_states,\n    dt_obs,\n    parameters,\n    var_obs,\n    nb_loop_train,\n    nb_loop_test,\n    sigma2_catalog,\n    sigma2_obs,\n)","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"compute u_0 to be in the attractor space","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"u0 = [8.0; 0.0; 30.0]\ntspan = (0.0, 5.0)\nprob = ODEProblem(ssm.model, u0, tspan, parameters)\nu0 = last(solve(prob, save_everystep = false))\n\nxt, yo, catalog = generate_data(ssm, u0);\n\nplot(xt.t, vcat(xt.u'...)[:, 1])\nscatter!(yo.t, vcat(yo.u'...)[:, 1]; markersize = 2)","category":"page"},{"location":"generated/lorenz63/#Classical-data-assimilation","page":"Lorenz 63","title":"Classical data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"regression = :local_linear\nsampling = :gaussian\nk, np = 100, 500\n\nDA = DataAssimilation(ssm, xt)\nx̂_classical = forecast(DA, yo, AnEnKS(np), progress = false)\n@time RMSE(xt, x̂_classical)","category":"page"},{"location":"generated/lorenz63/#Analog-data-assimilation","page":"Lorenz 63","title":"Analog data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"f = AnalogForecasting(k, xt, catalog; regression = regression, sampling = sampling)\nDA = DataAssimilation(f, xt, ssm.sigma2_obs)\n@time x̂_analog = forecast(DA, yo, AnEnKS(np), progress = false)\nprintln(RMSE(xt, x̂_analog))","category":"page"},{"location":"generated/lorenz63/#Comparison-between-classical-and-analog-data-assimilation","page":"Lorenz 63","title":"Comparison between classical and analog data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"plot(xt.t, xt[1], label = \"true state\")\nplot!(xt.t, x̂_classical[1], label = \"classical\")\nplot!(xt.t, x̂_analog[1], label = \"analog\")\nscatter!(yo.t, yo[1]; markersize = 2, label = \"observations\")","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"The results show that performances of the data-driven analog data assimilation are closed to those of the model-driven data assimilation. The error can be reduced by augmenting the size of the catalog \"nblooptrain\".","category":"page"},{"location":"generated/lorenz63/#Remark","page":"Lorenz 63","title":"Remark","text":"","category":"section"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"Note that for all the previous experiments, we use the robust Ensemble Kalman Smoother (EnKS) with the increment or local linear regressions and the Gaussian sampling. If you want to have realistic state estimations, we preconize the use of the Particle Filter PF with the locally constant regression (regression = :locally_constant) and the multinomial sampler (sampling = :multinomial) with a large number of particles np. For more details about the different options, see the attached publication: Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. Monthly Weather Review, 145(10), 4093-4107.","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"","category":"page"},{"location":"generated/lorenz63/","page":"Lorenz 63","title":"Lorenz 63","text":"This page was generated using Literate.jl.","category":"page"},{"location":"ensemble_kalman_filters/#Ensemble-filters","page":"Ensemble Kalman filters","title":"Ensemble filters","text":"","category":"section"},{"location":"ensemble_kalman_filters/","page":"Ensemble Kalman filters","title":"Ensemble Kalman filters","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"ensemble_kalman_filters.jl\"]","category":"page"},{"location":"ensemble_kalman_filters/#AnalogDataAssimilation.AnEnKF","page":"Ensemble Kalman filters","title":"AnalogDataAssimilation.AnEnKF","text":"EnKF( np )\n\nEnsemble Kalman Filters\n\n\n\n\n\n","category":"type"},{"location":"ensemble_kalman_filters/#AnalogDataAssimilation.forecast-Tuple{DataAssimilation, TimeSeries, AnEnKF}","page":"Ensemble Kalman filters","title":"AnalogDataAssimilation.forecast","text":"forecast( da, yo, mc; progress=true)\n\nda: DataAssimilation\nyo: Observations\nmc: Monte Carlo method\nprogress: display the progress bar.\n\nApply stochastic and sequential data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"EditURL = \"https://github.com/pnavaro/AnalogDataAssimilation.jl/blob/master/docs/examples/model_forecasting.jl\"","category":"page"},{"location":"generated/model_forecasting/#Model-Forecasting","page":"Model Forecasting","title":"Model Forecasting","text":"","category":"section"},{"location":"generated/model_forecasting/#Set-the-State-Space-model","page":"Model Forecasting","title":"Set the State-Space model","text":"","category":"section"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"using Plots\nusing AnalogDataAssimilation\nusing DifferentialEquations\n\nσ = 10.0\nρ = 28.0\nβ = 8.0 / 3\n\ndt_integration = 0.01\ndt_states = 1\ndt_obs = 8\nparameters = [σ, ρ, β]\nvar_obs = [1]\nnb_loop_train = 100\nnb_loop_test = 10\nsigma2_catalog = 0.0\nsigma2_obs = 2.0\n\nssm = StateSpaceModel(\n    lorenz63,\n    dt_integration,\n    dt_states,\n    dt_obs,\n    parameters,\n    var_obs,\n    nb_loop_train,\n    nb_loop_test,\n    sigma2_catalog,\n    sigma2_obs,\n)","category":"page"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"compute u0 to be in the attractor space","category":"page"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"u0 = [8.0; 0.0; 30.0]\ntspan = (0.0, 5.0)\nprob = ODEProblem(ssm.model, u0, tspan, parameters)\nu0 = last(solve(prob, save_everystep = false))","category":"page"},{"location":"generated/model_forecasting/#Generate-data","page":"Model Forecasting","title":"Generate data","text":"","category":"section"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"xt, yo, catalog = generate_data(ssm, u0);\n\nplot(xt.t, xt[1])\nscatter!(yo.t, yo[1]; markersize = 2)","category":"page"},{"location":"generated/model_forecasting/#Data-assimilation-with-model-forecasting","page":"Model Forecasting","title":"Data assimilation with model forecasting","text":"","category":"section"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"np = 100\nDA = DataAssimilation(ssm, xt)\n@time x̂ = forecast(DA, yo, AnPF(np), progress = false);\nprintln(RMSE(xt, x̂))","category":"page"},{"location":"generated/model_forecasting/#Plot-the-times-series","page":"Model Forecasting","title":"Plot the times series","text":"","category":"section"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"plot(xt.t, x̂[1])\nscatter!(xt.t, xt[1]; markersize = 2)\nplot!(xt.t, x̂[2])\nscatter!(xt.t, xt[2]; markersize = 2)\nplot!(xt.t, x̂[3])\nscatter!(xt.t, xt[3]; markersize = 2)","category":"page"},{"location":"generated/model_forecasting/#Plot-the-phase-space-plot","page":"Model Forecasting","title":"Plot the phase-space plot","text":"","category":"section"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"p = plot3d(\n    1,\n    xlim = (-25, 25),\n    ylim = (-25, 25),\n    zlim = (0, 50),\n    title = \"Lorenz 63\",\n    marker = 2,\n)\nfor x in eachrow(vcat(x̂.u'...))\n    push!(p, x...)\nend\np","category":"page"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"","category":"page"},{"location":"generated/model_forecasting/","page":"Model Forecasting","title":"Model Forecasting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"particle_filters/#Particle-filters","page":"Particle filters","title":"Particle filters","text":"","category":"section"},{"location":"particle_filters/","page":"Particle filters","title":"Particle filters","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"particle_filters.jl\"]","category":"page"},{"location":"particle_filters/#AnalogDataAssimilation.forecast-Tuple{DataAssimilation, TimeSeries, AnPF}","page":"Particle filters","title":"AnalogDataAssimilation.forecast","text":"data_assimilation( yo, da, AnPF(100) )\n\nApply particle filters data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"state-space/#State-Space","page":"State Space","title":"State Space","text":"","category":"section"},{"location":"state-space/","page":"State Space","title":"State Space","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"state_space.jl\"]","category":"page"},{"location":"state-space/#AnalogDataAssimilation.StateSpaceModel","page":"State Space","title":"AnalogDataAssimilation.StateSpaceModel","text":"Space-State model is defined through the following equations\n\nLet x_0 x_1 dots x_K  denote the discrete-time state process, which is typically unknown to the observer. Let y_1y_2 y_K  represent the observation process, which describes the measurements of the system that are available to the observer. \n\nThese state and observation processes are related following a nonlinear  state-space model: The first equation describes how the system state x_k  evolves through a nonlinear dynamical model f_k1k between successive  time steps t_k1 and t_k. η = η1η2 ηK is the model noise process, which accounts for the imperfections of the model; it is assumed to be independent and identically distributed. At each time step, η_k is assumed to be Gaussian with zero mean and covariance matrix Q. The second equation models how an observation y_k at time t_k is obtained from the state x_k through a linear operator H. ε = ε1 ε2     εk is an independent and identically distributed process representing the observation  errors; ε_k is assumed Gaussian with zero mean and covariance matrix R. Finally, the processes η and ε are assumed to be jointly independent and independent of the initial (background) state x0, which is assumed to be Gaussian with mean xb and covariance B.\n\nleft\nbeginarrayl\nX_t = m(X_t-1) + eta_t \nY_t = H(X_t) + varepsilon_t\nendarray\nright\n\nX : hidden variables\nY : observed variables\ndt_integrationis the numerical time step used to solve the ODE.\ndt_states is the number of dt_integration between X_t-1 and X_t.\ndt_obs is the number of dt_integration between Y_t-1 and Y_t.\n\n\n\n\n\n","category":"type"},{"location":"models/#Models","page":"Models","title":"Models","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"models.jl\"]","category":"page"},{"location":"models/#AnalogDataAssimilation.lorenz63-NTuple{4, Any}","page":"Models","title":"AnalogDataAssimilation.lorenz63","text":"lorenz63(du, u, p, t)\n\nLorenz-63 dynamical model u = x y z and p = sigma rho mu:\n\nfracdxdt = σ(y-x) \nfracdydt = x(ρ-z) - y \nfracdzdt = xy - βz \n\nExample Catalog\nLorenz system on wikipedia\n\n\n\n\n\n","category":"method"},{"location":"models/#AnalogDataAssimilation.lorenz96-NTuple{4, Any}","page":"Models","title":"AnalogDataAssimilation.lorenz96","text":"lorenz96(S, t, F, J)\n\nLorenz-96 dynamical model. For i=1N:\n\nfracdx_idt = (x_i+1-x_i-2)x_i-1 - x_i + F\n\nwhere it is assumed that x_-1=x_N-1x_0=x_N and x_N+1=x_1.  Here x_i is the state of the system and F is a forcing constant. \n\nLorenz 96 model on wikipedia\n\n\n\n\n\n","category":"method"},{"location":"ensemble_kalman_smoothers/#Ensemble-Kalman-smoothers","page":"Ensemble Kalman smoothers","title":"Ensemble Kalman smoothers","text":"","category":"section"},{"location":"ensemble_kalman_smoothers/","page":"Ensemble Kalman smoothers","title":"Ensemble Kalman smoothers","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"ensemble_kalman_smoothers.jl\"]","category":"page"},{"location":"ensemble_kalman_smoothers/#AnalogDataAssimilation.forecast-Tuple{DataAssimilation, TimeSeries, AnEnKS}","page":"Ensemble Kalman smoothers","title":"AnalogDataAssimilation.forecast","text":"data_assimilation( yo, da)\n\nApply stochastic and sequential data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"utils/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utils/","page":"Utilities","title":"Utilities","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"utils.jl\"]","category":"page"},{"location":"utils/#AnalogDataAssimilation.RMSE-Tuple{Any}","page":"Utilities","title":"AnalogDataAssimilation.RMSE","text":"RMSE(e)\n\nReturns the Root Mean Squared Error\n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.ensure_pos_sym-Union{Tuple{Matrix{T}}, Tuple{T}} where T<:AbstractFloat","page":"Utilities","title":"AnalogDataAssimilation.ensure_pos_sym","text":"ensure_pos_sym(M; ϵ= 1e-8)\n\nEnsure that matrix M is positive and symmetric to avoid numerical errors when numbers are small by doing (M + M')/2 + ϵ*I\n\nreference : StateSpaceModels.jl\n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.inv_using_SVD-Tuple{Any, Any}","page":"Utilities","title":"AnalogDataAssimilation.inv_using_SVD","text":"inv_using_SVD(Mat, eigvalMax)\n\nSVD decomposition of Matrix. \n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.mk_stochastic!-Tuple{Matrix{Float64}}","page":"Utilities","title":"AnalogDataAssimilation.mk_stochastic!","text":"mk_stochastic!(w)\n\nEnsure the matrix is stochastic, i.e.,  the sum over the last dimension is 1.\n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.normalise!-Tuple{Any}","page":"Utilities","title":"AnalogDataAssimilation.normalise!","text":"normalise!( w )\n\nNormalize the entries of a multidimensional array sum to 1.\n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.resample!-Tuple{Vector{Int64}, Vector{Float64}}","page":"Utilities","title":"AnalogDataAssimilation.resample!","text":"resample!( indx, w )\n\nMultinomial resampler.\n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.resample_multinomial-Tuple{Vector{Float64}}","page":"Utilities","title":"AnalogDataAssimilation.resample_multinomial","text":"resample_multinomial( w )\n\nMultinomial resampler. \n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.sample_discrete-Tuple{Any, Any, Any}","page":"Utilities","title":"AnalogDataAssimilation.sample_discrete","text":"sample_discrete(prob, r, c)\n\nSampling from a non-uniform distribution. \n\n\n\n\n\n","category":"method"},{"location":"utils/#AnalogDataAssimilation.sqrt_svd-Tuple{AbstractMatrix}","page":"Utilities","title":"AnalogDataAssimilation.sqrt_svd","text":"sqrt_svd(A)\n\nReturns the square root matrix by SVD\n\n\n\n\n\n","category":"method"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"EditURL = \"https://github.com/pnavaro/AnalogDataAssimilation.jl/blob/master/docs/examples/catalog.jl\"","category":"page"},{"location":"generated/catalog/#Example-Catalog","page":"Example Catalog","title":"Example Catalog","text":"","category":"section"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"using Plots, AnalogDataAssimilation\n\nσ = 10.0\nρ = 28.0\nβ = 8.0 / 3\n\ndt_integration = 0.01\ndt_states = 1\ndt_obs = 8\nparameters = [σ, ρ, β]\nvar_obs = [1]\nnb_loop_train = 100\nnb_loop_test = 10\nsigma2_catalog = 0.0\nsigma2_obs = 2.0\n\nssm = StateSpaceModel(\n    lorenz63,\n    dt_integration,\n    dt_states,\n    dt_obs,\n    parameters,\n    var_obs,\n    nb_loop_train,\n    nb_loop_test,\n    sigma2_catalog,\n    sigma2_obs,\n)\n\n\nxt, yo, catalog = generate_data(ssm, [10.0; 0.0; 0.0]);\nnothing #hide","category":"page"},{"location":"generated/catalog/#Time-series","page":"Example Catalog","title":"Time series","text":"","category":"section"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"plot(catalog.analogs[1, :])\nplot!(catalog.analogs[2, :])\nplot!(catalog.analogs[3, :])","category":"page"},{"location":"generated/catalog/#Phase-space-plot","page":"Example Catalog","title":"Phase space plot","text":"","category":"section"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"p = plot3d(\n    1,\n    xlim = (-25, 25),\n    ylim = (-25, 25),\n    zlim = (0, 50),\n    title = \"Lorenz 63\",\n    marker = 1,\n)\nfor x in eachcol(catalog.analogs)\n    push!(p, x...)\nend\np","category":"page"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"","category":"page"},{"location":"generated/catalog/","page":"Example Catalog","title":"Example Catalog","text":"This page was generated using Literate.jl.","category":"page"},{"location":"forecasting/#Forecasting","page":"Forecasting","title":"Forecasting","text":"","category":"section"},{"location":"forecasting/","page":"Forecasting","title":"Forecasting","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"model_forecasting.jl\", \"analog_forecasting.jl\"]","category":"page"},{"location":"forecasting/#AnalogDataAssimilation.StateSpaceModel-Tuple{Matrix{Float64}}","page":"Forecasting","title":"AnalogDataAssimilation.StateSpaceModel","text":"Apply the dynamical models to generate numerical forecasts.\n\n\n\n\n\n","category":"method"},{"location":"forecasting/#AnalogDataAssimilation.AnalogForecasting","page":"Forecasting","title":"AnalogDataAssimilation.AnalogForecasting","text":"AnalogForecasting(k, xt, catalog)\n\nparameters of the analog forecasting method\n\nk            : number of analogs\nneighborhood : global analogs\ncatalog      : catalog with analogs and successors\nregression   : (:locallyconstant, :increment, :locallinear)\nsampling     : (:gaussian, :multinomial)\n\n\n\n\n\n","category":"type"},{"location":"forecasting/#AnalogDataAssimilation.AnalogForecasting-Tuple{Matrix{Float64}}","page":"Forecasting","title":"AnalogDataAssimilation.AnalogForecasting","text":"Apply the analog method on catalog of historical data \nto generate forecasts.\n\n\n\n\n\n","category":"method"},{"location":"catalog/#Catalog","page":"Catalog","title":"Catalog","text":"","category":"section"},{"location":"catalog/","page":"Catalog","title":"Catalog","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"catalog.jl\", \"generate_data.jl\"]","category":"page"},{"location":"catalog/#AnalogDataAssimilation.Catalog","page":"Catalog","title":"AnalogDataAssimilation.Catalog","text":"Catalog( data, ssm)\n\nData type to store analogs and succesors observations from the Space State model\n\nExample Catalog\n\n\n\n\n\n","category":"type"},{"location":"catalog/#AnalogDataAssimilation.generate_data","page":"Catalog","title":"AnalogDataAssimilation.generate_data","text":"generate_data( ssm, u0; seed=42)\n\nfrom StateSpace generate:\n\ntrue state (xt)\npartial/noisy observations (yo)\ncatalog\n\n\n\n\n\n","category":"function"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"EditURL = \"https://github.com/pnavaro/AnalogDataAssimilation.jl/blob/master/docs/examples/analog_forecasting.jl\"","category":"page"},{"location":"generated/analog_forecasting/#Analog-forecasting","page":"Analog forecasting","title":"Analog forecasting","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"using Plots, DifferentialEquations, AnalogDataAssimilation\n\n\nσ = 10.0\nρ = 28.0\nβ = 8.0 / 3\n\ndt_integration = 0.01\ndt_states = 1\ndt_obs = 8\nvar_obs = [1]\nnb_loop_train = 100\nnb_loop_test = 10\nsigma2_catalog = 0.0\nsigma2_obs = 2.0\n\nssm = StateSpaceModel(\n    lorenz63,\n    dt_integration,\n    dt_states,\n    dt_obs,\n    [σ, ρ, β],\n    var_obs,\n    nb_loop_train,\n    nb_loop_test,\n    sigma2_catalog,\n    sigma2_obs,\n)","category":"page"},{"location":"generated/analog_forecasting/#compute-u0-to-be-in-the-attractor-space","page":"Analog forecasting","title":"compute u0 to be in the attractor space","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"u0 = [8.0; 0.0; 30.0]\ntspan = (0.0, 5.0)\nprob = ODEProblem(ssm.model, u0, tspan, ssm.params)\nu0 = last(solve(prob, reltol = 1e-6, save_everystep = false))","category":"page"},{"location":"generated/analog_forecasting/#Generate-the-data","page":"Analog forecasting","title":"Generate the data","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"xt, yo, catalog = generate_data(ssm, u0)","category":"page"},{"location":"generated/analog_forecasting/#Create-the-forecasting-function","page":"Analog forecasting","title":"Create the forecasting function","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"af = AnalogForecasting(50, xt, catalog; regression = :local_linear, sampling = :multinomial)","category":"page"},{"location":"generated/analog_forecasting/#Data-assimilation","page":"Analog forecasting","title":"Data assimilation","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"np = 100\nDA = DataAssimilation(af, xt, ssm.sigma2_obs)\nx̂ = forecast(DA, yo, AnEnKS(np), progress = false);\nprintln(RMSE(xt, x̂))","category":"page"},{"location":"generated/analog_forecasting/#Plot-results","page":"Analog forecasting","title":"Plot results","text":"","category":"section"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"plot(xt.t, xt[1], label = \"true\")\nplot!(xt.t, x̂[1], label = \"forecasted\")\nscatter!(yo.t, yo[1], markersize = 2, label = \"observed\")","category":"page"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"","category":"page"},{"location":"generated/analog_forecasting/","page":"Analog forecasting","title":"Analog forecasting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#AnalogDataAssimilation.jl","page":"Home","title":"AnalogDataAssimilation.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for AnalogDataAssimilation.jl","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"time-series/#Time-Series","page":"Time Series","title":"Time Series","text":"","category":"section"},{"location":"time-series/","page":"Time Series","title":"Time Series","text":"Modules = [AnalogDataAssimilation]\nPages   = [\"time_series.jl\"]","category":"page"},{"location":"time-series/#AnalogDataAssimilation.RMSE-Tuple{Any, Any}","page":"Time Series","title":"AnalogDataAssimilation.RMSE","text":"RMSE(a, b)\n\nCompute the Root Mean Square Error between 2 time series.\n\n\n\n\n\n","category":"method"},{"location":"time-series/#AnalogDataAssimilation.train_test_split-Tuple{TimeSeries, TimeSeries}","page":"Time Series","title":"AnalogDataAssimilation.train_test_split","text":"train_test_split( X, Y; test_size)\n\nSplit time series into random train and test subsets\n\n\n\n\n\n","category":"method"}]
}