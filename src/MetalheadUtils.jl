module MetalheadUtils

using Images

export imgFromURL, postprocess, normalize!
# Write your package code here.

function imgFromURL(url)
	mktemp() do fn,f
		download(url, fn)
    load(fn)
	end
end

## the reverse of Metalhead.preprocess
function postprocess(x)
	μ = [0.485, 0.456, 0.406]
	σ = [0.229, 0.224, 0.225]
	xx = permutedims(x[:,:,:,1], (3, 2, 1))./255
	xxx = N0f8.(xx.*σ .+ μ)
	[RGB(xxx[1,i,j], xxx[2,i,j], xxx[3,i,j]) for i = 1:size(xxx,2), j = 1:size(xxx,3)]
end

function normalize!(x, lower, upper)
	x .= min.(x, upper)
	x .= max.(x, lower)
	x
end



end
