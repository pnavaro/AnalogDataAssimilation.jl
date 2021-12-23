module AnalogDataAssimilation

include("utils.jl")
include("models.jl")
include("time_series.jl")
include("state_space.jl")
include("catalog.jl")
include("generate_data.jl")
include("data_assimilation.jl")
include("model_forecasting.jl")
include("local_linear.jl")
include("analog_forecasting.jl")

export forecast

include("ensemble_kalman_filters.jl")
include("ensemble_kalman_smoothers.jl")
include("particle_filters.jl")

end
