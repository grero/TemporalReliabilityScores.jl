module TemporalReliabilityScores
using Random
using StatsBase

function tr_entropy(X::Matrix{T}) where T <: Real
	nbins,ntrials = size(X)
	pp = fill(0.0, nbins)
	for tt in 1:ntrials
		ii = argmax(X[:,tt])
		pp[ii] += 1.0
	end
	pp ./= ntrials
	entropy(pp)
end

function StatsBase.entropy(X::BitMatrix)
	entropy(get_prob(X))
end

StatsBase.renyientropy(X::BitMatrix, α::Real) = renyientropy(get_prob(X), α)

"""
Get the probabilities of each row of `X`.
"""
function get_prob(X::BitMatrix)
	nb,nt = size(X)
	bb = (2).^[0:nb-1;]
	y = vec(bb'*X)
	[n/nt for n in values(StatsBase.countmap(y))]
end

"""
Compute the TREntropyScore for the rows of matrix `X`. The renyi entropy with parameter `
α` is computed for the distribution of binary rows, and this is compared to a shuffled surrogates
where for each trial the rows are shuffled randomly.
"""
function tr_entropy_score(X::BitMatrix, α::Real;nshuffles=1000,RNG=MersenneTwister(rand(UInt32)))
	nbins, ntrials = size(X)
	ee0 = fill(0.0, nshuffles)
	Xs = fill!(similar(X), false)
	for i in 1:nshuffles
		fill!(Xs, false)
		for j in 1:ntrials
			Xs[:,j] .= X[shuffle(RNG, 1:nbins), j]
		end
		ee0[i] = renyientropy(Xs,α)
	end
	ee1 = renyientropy(X,α)
	ii = searchsortedlast(sort(ee0), ee1)
	ee1, ii
end

"""
Generate a test care for temporal reliability, where `p` denotes the probability that a bin is active.
"""
function test_case(;p=[0.05,0.05,0.9,0.05,0.05],ntrials=20,RNG=MersenneTwister(rand(UInt32)))
	nbins = length(p)
	X = falses(nbins,ntrials)
	for j in 1:ntrials
		for i in 1:nbins
			X[i,j] = rand(RNG) < p[i]
		end
	end
	X
end
end # module
