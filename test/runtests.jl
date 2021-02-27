using MetalheadUtils
using Test
using Images, Metalhead

image_urls = ["https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg", "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg",
"https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
"https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
"https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
"https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"];

@testset "MetalheadUtils.jl" begin
    images = imgFromURL.(image_urls)
    @test length(images) == length(image_urls)
    @test images[1] isa Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}

    img1 = postprocess(Metalhead.preprocess(images[1]))
    @test img1 isa Array{RGB{FixedPointNumbers.Normed{UInt8,8}},2}
    r = img1 - impreprocess(images[1])
    @test maximum(abs.(channelview(r))) <= 1/500
end

using Flux
@testset "Features" begin
    r = Chain(x->x+1, x->x+2)
    @test Features(r, 20, [1]) == [21]
    @test Features(r, 20, [1,2]) == [21,23]

    vgg = VGG19()
    img1 = Metalhead.preprocess(imgFromURL(image_urls[1]))
    @test Features(vgg, img1, [1,2]) == [Features(vgg, img1, [1])[1], Features(vgg, img1, [2])[1]]
end
