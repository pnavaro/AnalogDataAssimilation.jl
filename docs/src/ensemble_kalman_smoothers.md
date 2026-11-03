# Ensemble Kalman smoothers

```@autodocs
Modules = [AnalogDataAssimilation]
Pages   = ["ensemble_kalman_smoothers.jl"]
```

`AnEnKS(np)` requires at least two ensemble members. Missing observation
components may be marked with `NaN`; infinite values are rejected. Set
`using Random` and pass `rng=MersenneTwister(seed)` to `forecast` to make its
sampling reproducible.
