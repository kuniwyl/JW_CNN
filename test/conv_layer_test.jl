function conv_layer_test() 
    @testset "Convolutional Layer Forward Tests" begin
        @testset "Forward Pass Test" begin
            input = ones(28, 28, 1);
            layer = ConvLayer(ones(3, 3, 1, 6), ones(6), 1, 0)
            output = forward_pass(layer, input)
            @test size(output) == (26, 26, 6)  # Check output dimensions
            @test output[1, 1, 1] == 10  # Check the first element of the output

            layer = ConvLayer(ones(3, 3, 6, 16), ones(16), 1, 0)
            output = forward_pass(layer, output)
            @test size(output) == (24, 24, 16)  # Check output dimensions
            @test output[1, 1, 1] == 90 * 6 + 1  # Check the first element of the output
        end
    end
end


# Example utility to create a sample convolutional layer
function create_sample_conv_layer()
    weights = rand(3, 3, 1, 2)  # 3x3 filters, 1 input channel, 2 output channels
    bias = zeros(2)             # No bias for simplicity
    stride = 1
    padding = 0
    return ConvLayer(weights, bias, stride, padding)
end

# Function to generate predictable inputs and outputs
function generate_data(input_dim, num_channels, batch_size)
    return rand(input_dim, input_dim, num_channels, batch_size)  # Input dimensions and batch size
end

