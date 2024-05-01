function dense_layer_test()
# Test the DenseLayer functionality
@testset "DenseLayer Tests" begin
    @testset "Forward Pass Test" begin
        input = ones(10)
        layer = DenseLayer(ones(10, 5), ones(5), relu)
        output = forward_pass(layer, input)
        @test size(output) == (5,)  #Check output dimensions
        @test output[1] == 11  # Check the first element of the output

        layer = DenseLayer(ones(5, 2), ones(2), relu)
        output = forward_pass(layer, output)
        @test size(output) == (2,)  # Check output dimensions
        @test output[1] == 56  # Check the first element of the output
    end
end

end

function relu(z)
    return max.(0, z)
end