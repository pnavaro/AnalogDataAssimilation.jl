@testset "EM covariance estimation" begin
    using LinearAlgebra
    using Random

    observations = [
        0.2 0.1 NaN 0.4
        -0.1 NaN NaN 0.3
    ]
    initial_state = [0.0, 0.0]
    initial_covariance = Matrix{Float64}(I, 2, 2)
    model_noise_covariance = 0.1 .* Matrix{Float64}(I, 2, 2)
    observation_noise_covariance = 0.2 .* Matrix{Float64}(I, 2, 2)
    dynamics(x) = [0.9x[1] + 0.1x[2], -0.1x[1] + 0.8x[2]]
    dynamics_jacobian(x) = [0.9 0.1; -0.1 0.8]
    observation_operator(x) = copy(x)
    observation_jacobian(x) = Matrix{Float64}(I, 2, 2)

    @testset "Extended Kalman smoother EM" begin
        result = em_eks(
            observations;
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            dynamics_jacobian = dynamics_jacobian,
            observation_operator = observation_operator,
            observation_jacobian = observation_jacobian,
            iterations = 2,
            model_noise_structure = :diag,
            truth = zeros(2, 5),
        )

        @test result isa CovarianceEMResult
        @test result.algorithm === :eks
        @test size(result.smoothed_states) == (2, 5, 2)
        @test size(result.smoothed_covariances) == (2, 2, 5, 2)
        @test size(result.model_noise_covariance) == (2, 2, 3)
        @test size(result.observation_noise_covariance) == (2, 2, 3)
        @test size(result.background_state) == (2, 3)
        @test all(isfinite, result.loglikelihood)
        @test all(isfinite, result.rmse)
        @test all(isapprox.(result.smoothed_covariances[1, 2, :, :], result.smoothed_covariances[2, 1, :, :]; atol = 1e-12))
        @test all(minimum(eigvals(Symmetric(result.smoothed_covariances[:, :, t, i]))) >= -1e-10 for t in 1:5, i in 1:2)
        @test all(isapprox.(result.model_noise_covariance[1, 2, :], 0.0; atol = 1e-12))
        @test all(isapprox.(result.model_noise_covariance[2, 1, :], 0.0; atol = 1e-12))
        @test all(isposdef(Symmetric(result.observation_noise_covariance[:, :, i])) for i in 1:3)
    end

    @testset "Ensemble Kalman smoother EM" begin
        options = (
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            observation_matrix = Matrix{Float64}(I, 2, 2),
            ensemble_size = 20,
            iterations = 2,
            model_noise_structure = :const,
            model_noise_template = [1.0 0.2; 0.2 2.0],
        )
        result = em_enks(
            observations;
            options...,
            rng = MersenneTwister(17),
        )
        repeat_result = em_enks(
            observations;
            options...,
            rng = MersenneTwister(17),
        )

        @test result.algorithm === :enks
        @test size(result.smoothed_states) == (2, 5, 2)
        @test size(result.smoothed_covariances) == (2, 2, 5, 2)
        @test all(isfinite, result.loglikelihood)
        @test result.smoothed_states == repeat_result.smoothed_states
        @test result.smoothed_covariances == repeat_result.smoothed_covariances
        @test result.model_noise_covariance == repeat_result.model_noise_covariance
        for i in 2:3
            Q = result.model_noise_covariance[:, :, i]
            @test isapprox(Q[1, 2] / Q[1, 1], 0.2; atol = 1e-10)
            @test isapprox(Q[2, 1] / Q[1, 1], 0.2; atol = 1e-10)
            @test isposdef(Symmetric(result.observation_noise_covariance[:, :, i]))
        end
    end

    @testset "Validation and fixed parameters" begin
        fixed = em_eks(
            observations;
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            dynamics_jacobian = dynamics_jacobian,
            observation_operator = observation_operator,
            observation_jacobian = observation_jacobian,
            iterations = 1,
            estimate_model_noise = false,
            estimate_observation_noise = false,
            estimate_background = false,
        )
        @test fixed.model_noise_covariance[:, :, 1] == fixed.model_noise_covariance[:, :, 2]
        @test fixed.observation_noise_covariance[:, :, 1] == fixed.observation_noise_covariance[:, :, 2]
        @test fixed.background_state[:, 1] == fixed.background_state[:, 2]

        @test_throws ArgumentError em_eks(
            fill(Inf, 2, 4);
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            dynamics_jacobian = dynamics_jacobian,
            observation_jacobian = observation_jacobian,
        )
        @test_throws DimensionMismatch em_enks(
            observations;
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            observation_matrix = ones(1, 1),
        )
        @test_throws ArgumentError em_enks(
            observations;
            initial_state = initial_state,
            initial_covariance = initial_covariance,
            model_noise_covariance = model_noise_covariance,
            observation_noise_covariance = observation_noise_covariance,
            dynamics = dynamics,
            observation_matrix = Matrix{Float64}(I, 2, 2),
            ensemble_size = 1,
        )
    end
end
