using Test
using JW_CNN

include("conv_layer_test.jl")
conv_layer_test()

include("dense_layer_test.jl")
dense_layer_test()

include("pool_layer_test.jl")
pool_layer_test()

include("flatten_layer_test.jl")
flatten_layer_test()