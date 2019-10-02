using TemporalReliabilityScores
using Random
const TSR = TemporalReliabilityScores
using Test

@testset "Test case" begin
	RNG = MersenneTwister(1234)
	X = TSR.test_case(RNG=RNG)
	@test X[:,1] == [false, false, true, false, false]
	@test dropdims(sum(X,dims=2),dims=2) == [2; 1; 19; 2; 2]
end

@testset "TREntropy" begin
	X = permutedims(repeat([0 0 1 0], 10, 1), [2,1])
	te = TSR.tr_entropy(X)
	@test te ≈ 0.0
	X[:,5] .= [0,1,0,0]
	te = TSR.tr_entropy(X)
	@test te ≈ 0.3250829733914482
end

@testset "Full entropy" begin
	RNG = MersenneTwister(1234)
	idx1 = fill(3,20)
	idx2 = [3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 4, 3, 3, 3]
	idx3 = [3, 3, 4, 2, 2, 3, 4, 2, 4, 2, 3, 3, 4, 4, 4, 4, 3, 3, 4, 3]
	idx4 = [2, 3, 1, 3, 5, 5, 4, 2, 2, 1, 1, 2, 2, 1, 1, 3, 4, 3, 2, 4]
	X = falses(5,20)
	ee = fill(0.0, 4)
	eer = fill(0.0, 4)
	score = fill(0.0, 4)
	for (kk,idx) in enumerate([idx1,idx2,idx3,idx4])
		fill!(X, false)
		for (ii,_idx) in enumerate(idx)
			X[_idx,ii] = true
		end
		ee[kk] = TSR.entropy(X)
		eer[kk],score[kk] = TSR.tr_entropy_score(X,2.0,RNG=RNG)
	end
	@test ee[1] ≈ -0.0
	@test eer[1] ≈ -0.0
	@test score[1] ≈ 0.0
	@test ee[2] ≈ 0.6108643020548935
	@test eer[2] ≈ 0.5447271754416722
	@test score[2] ≈ 0.0
	@test ee[3] ≈ 1.0549201679861442
	@test eer[3] ≈ 1.021651247531981
	@test score[3] ≈ 4.0
	@test ee[4] ≈ 1.5444795210968603
	@test eer[4] ≈ 1.491654876777717
	@test score[4] ≈ 715.0
end

@testset "Temporal entropy score" begin
	idx1 = fill(3,20)
	idx2 = [3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 4, 3, 3, 3]
	X = falses(5,20)
	for (ii, _idx) in enumerate(idx2)
		X[_idx,ii] = true
	end
	ee1,ee0 = TSR.tr_entropy_score(X, 2.0;RNG=MersenneTwister(1234))
	@test ee1 ≈ 0.5447271754416722
	@test ee0 ≈ 1.436004650013498
end
