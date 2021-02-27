module MetalheadUtils

using Images, Metalhead, Flux

export imgFromURL, normalize!
export postprocess, impreprocess, vggBlockLen
export Features


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

## from Metalhead.preprocess
function impreprocess(im)
    # Resize such that smallest edge is 256 pixels long
    im = Metalhead.resize_smallest_dimension(im, 256)
    # Center-crop to 224x224
    im = Metalhead.center_crop(im, 224)
    im
end

const vggBlockLen = [3, 3, 5, 5, 5]

function Features(chain::Chain, X, layernums)
	[chain[1:k](X) for k in layernums]
end	

function Features(nn, X, layernums)
	Features(nn.layers, X, layernums)
end

end
