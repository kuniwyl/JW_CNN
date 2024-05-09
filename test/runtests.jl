using Test
using Flux
using JW_CNN

include("layers_unit/ConvLayerTest.jl")
include("layers_unit/MaxPoolLayerTest.jl")
include("layers_unit/FlattenLayerTest.jl")
include("layers_unit/FCLayerTest.jl")

function run_tests()
    ConvLayerTest_ForwardTest()
    MaxPoolLaterTest_ForwardTest()
    FlattenLayerTest_ForwardTest()
    FCLayerTest_ForwardTest()

    FCLayerTest_BackwardTest()
    FlattenLayerTest_BackwardTest()
    MaxPoolLayerTest_BackwardTest()
    ConvLayerTest_BackwardTest()

    FCLayerTest_UpdateWeights()
    ConvLayerTest_UpdateWeight()
end

run_tests()