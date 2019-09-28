using TemporalReliabilityScores
const TSR = TemporalReliabilityScores
using Test

@testset "TREntropy" begin
	X = permutedims(repeat([0 0 1 0], 10, 1), [2,1])
	te = TSR.tr_entropy(X)
	@test te ≈ 0.0
	X[:,5] .= [0,1,0,0]
	te = TSR.tr_entropy(X)
	@test te ≈ 0.3250829733914482
end
