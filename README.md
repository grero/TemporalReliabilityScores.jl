# TemporalReliabilityScores
[![Build Status](https://travis-ci.org/grero/TemporalReliabilityScores.jl.svg?branch=master)](https://travis-ci.org/grero/TemporalReliabilityScores.jl)

## Usage

```julia
using TemporalReliabilityScores
X = rand(5,50)
te = TemporalReliabilityScores.tr_entropy(X)
```
