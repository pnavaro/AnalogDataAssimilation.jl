using LinearAlgebra
using Random
using Test
import AnalogDataAssimilation: resample!

struct IdentityEnsembleForecast <: AnalogDataAssimilation.AbstractForecasting end
(::IdentityEnsembleForecast)(ensemble::Matrix{Float64}) = (copy(ensemble), copy(ensemble))

function assimilation_case(observations; prior = zeros(length(first(observations))), variance = 0.1, noise = 0.01)
    nt, nv = length(observations), length(first(observations))
    truth = TimeSeries(collect(0.0:(nt - 1)), [zeros(nv) for _ in 1:nt])
    yo = TimeSeries(collect(0.0:(nt - 1)), [Float64.(observation) for observation in observations])
    da = DataAssimilation(IdentityEnsembleForecast(), truth, noise)
    da.xb .= prior
    da.B .= variance .* Matrix{Float64}(I, nv, nv)
    return da, yo
end

@testset "Ensemble filters and particle filter regressions" begin
    @test_throws ArgumentError AnEnKF(1)
    @test_throws ArgumentError AnEnKS(1)
    @test_throws ArgumentError AnPF(1)

    @testset "Observation validation and missing values" begin
        da, yo = assimilation_case([[NaN, 1.0], [NaN, NaN]])
        for filter in (AnEnKF(20), AnEnKS(20), AnPF(20))
            estimate = forecast(da, yo, filter; progress = false, rng = MersenneTwister(8))
            @test all(all(isfinite, state) for state in estimate.u)
        end

        da, bad_observations = assimilation_case([[Inf, 1.0]])
        for filter in (AnEnKF(20), AnEnKS(20), AnPF(20))
            @test_throws ArgumentError forecast(
                da, bad_observations, filter; progress = false, rng = MersenneTwister(8),
            )
        end
    end

    @testset "AnEnKS smoothing covariance normalization" begin
        da, yo = assimilation_case([[NaN], [1.0]])
        da.R .= 0.01
        estimate = forecast(
            da, yo, AnEnKS(20_000); progress = false, rng = MersenneTwister(18),
        )
        @test isapprox(estimate.u[1][1], estimate.u[2][1]; atol = 0.04)
        repeated = forecast(
            da, yo, AnEnKS(20_000); progress = false, rng = MersenneTwister(18),
        )
        @test estimate.u == repeated.u
    end

    @testset "Particle filter posterior weighting" begin
        da, yo = assimilation_case([[0.2]])
        da.R .= 0.01
        estimate = forecast(
            da, yo, AnPF(100_000); progress = false, rng = MersenneTwister(23),
        )
        kalman_mean = 0.1 / (0.1 + 0.01) * 0.2
        @test isapprox(estimate.u[1][1], kalman_mean; atol = 0.006)

        repeated = forecast(
            da, yo, AnPF(100_000); progress = false, rng = MersenneTwister(23),
        )
        @test estimate.u == repeated.u
    end

    @testset "Particle filter first and all observations missing" begin
        da, yo = assimilation_case([[NaN], [NaN], [2.0]]; prior = [2.0])
        estimate = forecast(
            da, yo, AnPF(2_000); progress = false, rng = MersenneTwister(32),
        )
        @test all(all(isfinite, state) for state in estimate.u)
        @test all(isapprox(state[1], 2.0; atol = 0.1) for state in estimate.u[1:2])

        da, yo = assimilation_case([[NaN], [NaN], [NaN]]; prior = [2.0])
        estimate = forecast(
            da, yo, AnPF(2_000); progress = false, rng = MersenneTwister(33),
        )
        @test all(isapprox(state[1], 2.0; atol = 0.1) for state in estimate.u)
    end

    @testset "Particle filter stable and degenerate likelihoods" begin
        da, yo = assimilation_case([[1.0e6]])
        estimate = forecast(
            da, yo, AnPF(2_000); progress = false, rng = MersenneTwister(44),
        )
        @test all(isfinite, estimate.u[1])

        da, yo = assimilation_case([[1.0e200]])
        @test_throws ArgumentError forecast(
            da, yo, AnPF(2_000); progress = false, rng = MersenneTwister(44),
        )
    end

    @testset "Multinomial resampling validation and CDF boundary" begin
        indices = zeros(Int64, 100)
        resample!(indices, [0.0, 1.0, 0.0]; rng = MersenneTwister(51))
        @test all(==(2), indices)
        @test_throws ArgumentError resample!(indices, [0.0, 0.0])
        @test_throws ArgumentError resample!(indices, [1.0, NaN])
        @test_throws ArgumentError resample!(indices, [1.0, Inf])
        @test_throws ArgumentError resample!(indices, [1.0, -0.5])
        one_index = zeros(Int64, 1)
        @test resample!(one_index, [1.0, 0.0]) == [1]
        resample!(indices, [0.2, 0.2, 0.2])
        @test all(index -> index in 1:3, indices)
    end

    @testset "Filter RNG reaches stochastic analog forecasts" begin
        model = StateSpaceModel(
            (du, u, p, t) -> nothing,
            1.0,
            1,
            1,
            Float64[],
            [1],
            1,
            1,
            0.0,
            0.01,
        )
        times = [0.0, 1.0, 2.0, 3.0]
        truth = TimeSeries(times, [[0.0], [1.0], [2.0], [4.0]])
        observations = TimeSeries(times, [fill(NaN, 1) for _ in times])
        catalog = Catalog(reshape([0.0, 1.0, 2.0, 4.0], 1, :), model)
        for sampling in (:multinomial, :gaussian)
            analog = AnalogForecasting(
                2,
                truth,
                catalog;
                regression = :locally_constant,
                sampling = sampling,
            )
            da = DataAssimilation(analog, truth, 0.01)
            for filter in (AnEnKF(32), AnEnKS(32), AnPF(32))
                first_run = forecast(
                    da, observations, filter; progress = false, rng = MersenneTwister(91),
                )
                repeated_run = forecast(
                    da, observations, filter; progress = false, rng = MersenneTwister(91),
                )
                @test first_run.u == repeated_run.u
            end
        end
    end
end
