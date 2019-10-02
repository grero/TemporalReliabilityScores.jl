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
	#check that successive randomness with more than one bin active is tracked by the score
	RNG = MersenneTwister(1234)
	X = TSR.test_case(RNG=RNG)
	ee1,ii = TSR.tr_entropy_score(X, 2.0;RNG=RNG)
	@test ee1 ≈ 0.809680996815897
	@test ii == 0
	X = TSR.test_case(p=[0.1,0.1,0.9,0.1,0.1],RNG=RNG)
	ee1,ii = TSR.tr_entropy_score(X, 2.0;RNG=RNG)
	@test ee1 ≈ 1.0788096613719298
	@test ii == 0
	X = TSR.test_case(p=[0.3,0.3,0.9,0.3,0.3],RNG=RNG)
	ee1,ii = TSR.tr_entropy_score(X, 2.0;RNG=RNG)
	@test ee1 ≈ 1.9661128563728327
	@test ii == 1
	X = TSR.test_case(p=[0.5,0.5,0.9,0.5,0.5],RNG=RNG)
	ee1,ii = TSR.tr_entropy_score(X, 2.0;RNG=RNG)
	@test ee1 ≈ 2.353878387381596
	@test ii == 172
	X = TSR.test_case(p=[0.7,0.7,0.9,0.7,0.7],RNG=RNG)
	ee1,ii = TSR.tr_entropy_score(X, 2.0;RNG=RNG)
	@test ee1 ≈ 1.7429693050586228
	@test ii == 327
end
