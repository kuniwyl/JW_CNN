include("conv_help.jl")

function conv_layer_test() 
    @testset "Convolutional Layer Forward Tests" begin
        input = ones(28, 28, 1);
        layer = ConvLayer(ones(3, 3, 1, 6), ones(6))
        output = forward_pass(layer, input)
        @test size(output) == (26, 26, 6)  # Check output dimensions
        @test output[1, 1, 1] == 10  # Check the first element of the output

        layer = ConvLayer(ones(3, 3, 6, 16), ones(16))
        output = forward_pass(layer, output)
        @test size(output) == (24, 24, 16)  # Check output dimensions
        @test output[1, 1, 1] == 90 * 6 + 1  # Check the first element of the output
    end

    @testset "Convolutional Layer Backward Tests" begin
        input = input_data()
        filters = filters_values()
        biases = biases_values()
    
        layer = ConvLayer(filters, biases)
        output = forward_pass(layer, input)
        @test size(output) == (2, 2, 3)  # Check output dimensions
        @test isapprox(conv_output_forward(), output, atol = 1e-3)  # Check the first element of the output

        grad_output = gradient_from_next_layer_values()
        dL_dW, dL_dB = backward_pass(layer, grad_output, input)
        @test size(dL_dW) == size(filters)  # Check gradient dimensions
        @test size(dL_dB) == size(biases)  # Check gradient dimensions
        @test isapprox(gradient_for_filters_values(), dL_dW, atol = 1e-3)  # Check the first element of the gradient
        @test isapprox(gradient_for_biases_values(), dL_dB, atol = 1e-3)  # Check the first element of the gradient

        learning_rate = 0.01
        update_weights!(layer, dL_dW, dL_dB, learning_rate)
        @test isapprox(layer.weights, updated_filters(), atol = 1e-3)  # Check the first element of the updated weights
        @test isapprox(layer.biases, updated_biases(), atol = 1e-3)  # Check the first element of the updated biases
    end
end
