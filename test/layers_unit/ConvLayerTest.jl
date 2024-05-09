function ConvLayerTest_ForwardTest() 
    @testset "first layer example 1 -> 2" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 3, 3, 1, 1);

        conv = ConvLayer(2, 2, 2);
        filters = [1, 3, 3, 2, 4, 2, 1, 4];
        filters = reshape(filters, 2, 2, 2);
        conv.filters = filters;
        conv.biases = [1, 2];

        output = forward_pass(conv, input);
        expected = [50, 62, 57, 56, 59, 73, 54, 86];
        expected = reshape(expected, 2, 2, 2, 1);
        expected = convert(Array{Float32, 4}, expected);
        
        @test expected == output;
    end

    @testset "second layer example 2 -> 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 1);

        conv = ConvLayer(2, 2, 3);
        filters = [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1];
        filters = reshape(filters, 2, 2, 3);
        conv.filters = filters;
        conv.biases = [1, 2, 3];

        output = forward_pass(conv, input);
        expected = [751, 743, 500];
        expected = reshape(expected, 1, 1, 3, 1);
        expected = convert(Array{Float32, 4}, expected);

        @test expected == output;
    end

    @testset "third layer example batch_size 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 3);
        
        conv = ConvLayer(2, 2, 3);
        filters = [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1];
        filters = reshape(filters, 2, 2, 3);
        conv.filters = filters;
        conv.biases = [1, 2, 3];

        output = forward_pass(conv, input)
        expected = [751, 743, 500, 751, 743, 500, 751, 743, 500];
        expected = reshape(expected, 1, 1, 3, 3);
        expected = convert(Array{Float32, 4}, expected);

        @test expected == output;
    end
end

function ConvLayerTest_BackwardTest()
    @testset "first layer example 1 -> 2" begin
        input = [1, 2, 7, 3, 9, 1, 4, 8, 3];
        input = reshape(input, 3, 3, 1, 1);
        input = convert(Array{Float32, 4}, input);

        conv = ConvLayer(2, 2, 1);
        filters = [1, 2, 2, 1];
        filters = reshape(filters, 2, 2, 1);
        conv.filters = filters;
        conv.biases = [1];

        output = forward_pass(conv, input);
        
        gradients = [1, -1, 1, -1];
        gradients = reshape(gradients, 2, 2, 1, 1);
        gradients = convert(Array{Float32, 4}, gradients);

        dL_dIn = backward_pass(conv, gradients);

        expected_dL_dIn = [1, 1, -2, 3, 0, -3, 2, -1, -1];
        expected_dL_dIn = reshape(expected_dL_dIn, 3, 3, 1, 1);
        expected_dL_dIn = convert(Array{Float32, 4}, expected_dL_dIn);

        exprected_dL_dW = [-7, 3, -10, 13];
        exprected_dL_dW = reshape(exprected_dL_dW, 2, 2, 1);
        exprected_dL_dW = convert(Array{Float32, 3}, exprected_dL_dW);

        expected_dL_dB = [0];

        @show expected_dL_dIn

        @test expected_dL_dIn == dL_dIn;
        @test exprected_dL_dW == conv.dL_dW;
        @test expected_dL_dB == conv.dL_dB;
    end

    @testset "second layer example 2 -> 3" begin
        input = [1, 2, 7, 3, 9, 1, 4, 8, 3, 1, 2, 7, 3, 9, 1, 4, 8, 3];
        input = reshape(input, 3, 3, 1, 2);
        input = convert(Array{Float32, 4}, input);

        conv = ConvLayer(2, 2, 1);
        filters = [1, 2, 2, 1];
        filters = reshape(filters, 2, 2, 1);
        conv.filters = filters;
        conv.biases = [1];

        output = forward_pass(conv, input);
        
        gradients = [1, -1, 1, -1, 1, -1, 1, -1];
        gradients = reshape(gradients, 2, 2, 1, 2);
        gradients = convert(Array{Float32, 4}, gradients);

        dL_dIn = backward_pass(conv, gradients);

        expected_dL_dIn = [1, 1, -2, 3, 0, -3, 2, -1, -1, 1, 1, -2, 3, 0, -3, 2, -1, -1];
        expected_dL_dIn = reshape(expected_dL_dIn, 3, 3, 1, 2);
        expected_dL_dIn = convert(Array{Float32, 4}, expected_dL_dIn);

        exprected_dL_dW = [-7, 3, -10, 13] .* 2;
        exprected_dL_dW = reshape(exprected_dL_dW, 2, 2, 1);
        exprected_dL_dW = convert(Array{Float32, 3}, exprected_dL_dW);

        expected_dL_dB = [0];

        @test expected_dL_dIn == dL_dIn;
        @test exprected_dL_dW == conv.dL_dW;
        @test expected_dL_dB == conv.dL_dB;
    end

    @testset "third layer example batch_size 2 filter_size 2" begin
        input = [1, 2, 7, 3, 3, 6, 2, 5, 8, 1, 6, 5, 2, 1, 3, 3, 4, 2];
        input = reshape(input, 3, 3, 2, 1);
        input = convert(Array{Float32, 4}, input);

        conv = ConvLayer(2, 2, 3);
        filters = [1, 2, 2, 1, 1, 3, 3, 1, 1, 4, 4, 1];
        filters = reshape(filters, 2, 2, 3);
        conv.filters = filters;
        conv.biases = [1, 2, 3];

        output = forward_pass(conv, input);

        gradients = [1, 0, -1, 0, 0, -1, 0, 1, 1, 0, 0, 1];
        gradients = reshape(gradients, 2, 2, 3, 1);
        gradients = convert(Array{Float32, 4}, gradients);

        dL_dIn = backward_pass(conv, gradients);

        expected_dL_dIn = [2, 5, -3, 5, -1, 6, -2, 6, 2, 2, 5, -3, 5, -1, 6, -2, 6, 2];
        expected_dL_dIn = reshape(expected_dL_dIn, 3, 3, 2, 1);
        expected_dL_dIn = convert(Array{Float32, 4}, expected_dL_dIn);

        exprected_dL_dW = [-3, 4, 0, -5, -4, -3, 5, 1, 6, 17, 14, 14];
        exprected_dL_dW = reshape(exprected_dL_dW, 2, 2, 3);
        exprected_dL_dW = convert(Array{Float32, 3}, exprected_dL_dW);

        expected_dL_dB = [0, 0, 2];

        @test expected_dL_dIn == dL_dIn;
        @test exprected_dL_dW == conv.dL_dW;
        @test expected_dL_dB == conv.dL_dB;
    end
end

function ConvLayerTest_UpdateWeight()
    @testset "first layer example 1 -> 2" begin
        conv = ConvLayer(2, 2, 1);
        filters = [1, 2, 2, 1];
        filters = reshape(filters, 2, 2, 1);
        conv.filters = filters;
        conv.biases = [1];
        
        dL_dW = [1, 2, 3, 4];
        dL_dW = reshape(dL_dW, 2, 2, 1);
        dL_dW = convert(Array{Float32, 3}, dL_dW);
        conv.dL_dW = dL_dW;
        conv.dL_dB = [1];

        update_weights(conv, 0.1, 1);

        expected_filters = [0.9, 1.8, 1.7, 0.6];
        expected_filters = reshape(expected_filters, 2, 2, 1);
        expected_filters = convert(Array{Float32, 3}, expected_filters);

        expected_biases = [0.9];
        expected_biases = convert(Array{Float32, 1}, expected_biases);

        @test expected_filters == conv.filters;
        @test expected_biases == conv.biases;
    end
end