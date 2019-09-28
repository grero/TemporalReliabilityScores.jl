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
	nb,nt = size(X)
	bb = (2).^[0:nb-1;]
	y = vec(bb'*X)
	entropy([n/nt for n in values(StatsBase.countmap(y))])
end

end # module
