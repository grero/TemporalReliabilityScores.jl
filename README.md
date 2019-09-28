# TemporalReliabilityScores
[![Build Status](https://travis-ci.org/grero/TemporalReliabilityScores.jl.svg?branch=master)](https://travis-ci.org/grero/TemporalReliabilityScores.jl)
[![codecov](https://codecov.io/gh/grero/TemporalReliabilityScores.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/grero/TemporalReliabilityScores.jl)

## Usage

```julia
using TemporalReliabilityScores
X = rand(0:2, 5,50)
te = TemporalReliabilityScores.tr_entropy(X)
```
