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

@testset "Full entropy" begin
	idx1 = fill(3,20)
	idx2 = [3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 4, 3, 3, 3]
	idx3 = [3, 3, 4, 2, 2, 3, 4, 2, 4, 2, 3, 3, 4, 4, 4, 4, 3, 3, 4, 3]
	idx4 = [2, 3, 1, 3, 5, 5, 4, 2, 2, 1, 1, 2, 2, 1, 1, 3, 4, 3, 2, 4]
	X = falses(5,20)
	ee = fill(0.0, 4)
	for (kk,idx) in enumerate([idx1,idx2,idx3,idx4])
		fill!(X, false)
		for (ii,_idx) in enumerate(idx)
			X[_idx,ii] = true
		end
		ee[kk] = TSR.entropy(X)
	end
	@test ee[1] ≈ -0.0
	@test ee[2] ≈ 0.6108643020548935
	@test ee[3] ≈ 1.0549201679861442
	@test ee[4] ≈ 1.5444795210968603
end
