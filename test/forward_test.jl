include("./gen/bsize_case_full_test_gen.jl")

function forward_test()
    @testset "forward" begin
        @testset "batch_size=3" begin
            result, result1, filters1, filters2, matrix = bsize_case_full_test_gen(3)

            conv1 = ConvLayer(2, 3, 3)
            conv1.weights = filters1
            output = forward_pass(conv1, matrix)
            @show output
            @show result
            @show size(output) 
            @show size(result)
            @test output ≈ result

            conv2 = ConvLayer(3, 3, 3)
            conv2.weights = filters2
            output = forward_pass(conv2, result)
            @test output ≈ result1

            # pool = MaxPoolLayer(2, 2)
            # output = forward_pass(pool, result1)
            # @test output ≈ result2

            # flat = FlattenLayer()
            # output = forward_pass(flat, result2)
            # @test output ≈ result3

            # fc = FCLayer(12, 10)
            # fc.weights = weights
            # output = forward_pass(fc, result3)
            # @test output ≈ result4

            
        end
    end
end