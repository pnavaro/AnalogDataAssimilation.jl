# Particle filters

```@autodocs
Modules = [AnalogDataAssimilation]
Pages   = ["particle_filters.jl"]
```

`AnPF(np)` requires at least two particles. Missing observation components may
be marked with `NaN`; infinite values are rejected. Estimates are computed
from normalized importance weights before multinomial resampling, and
resampled particles receive equal weights. The filter retains the small
`da.B / 100` Gaussian perturbation after each forecast. Set
`using Random` and pass `rng=MersenneTwister(seed)` to `forecast` for
reproducible sampling.
