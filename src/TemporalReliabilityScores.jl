module TemporalReliabilityScores
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

end # module
