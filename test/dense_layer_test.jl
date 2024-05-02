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

    @testset "Backward Pass Test" begin
        inputs = [1.0, 2.0];
        weights = [0.5 -0.5; 1.5 -1.5];
        bias = [0.1, -0.1];
        true_output = [1.0, 0.0];
        learning_rate = 0.01;
        layer = DenseLayer(weights, bias, relu);
        
        y = forward_pass(layer, inputs);
        @test y == [3.6, 0.0]

        gradients = backward_pass(layer, y, inputs, true_output);
        @test gradients[1] ≈ [2.6 0.0; 5.2 0.0] 
        @test gradients[2] == [2.6, 0.0] 

        update_weights!(layer, gradients[1], gradients[2], learning_rate);
        @test layer.weights ≈ [0.474 -0.5; 1.448 -1.5]
        @test isapprox(layer.biases, [0.074, -0.1], atol=1e-3)
    end
end
end

function relu(z)
    return max.(0, z)
end

function relu_derivative(z)
    return z .> 0
end