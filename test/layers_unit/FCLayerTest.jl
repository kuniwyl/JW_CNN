function FCLayerTest_ForwardTest()
    @testset "first layer example 1 -> 2" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 9, 1);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(9, 2);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        weights = reshape(weights, 9, 2);
        fc.weights = weights;
        fc.biases = [1, 2];

        output = forward_pass!(fc, input);
        expected = [270, 712];
        expected = reshape(expected, 2, 1);
        expected = convert(Array{Float32, 2}, expected);
        
        @test expected == output;
    end

    @testset "first layer example 1 -> 3" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 9, 1);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(9, 3);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        weights = reshape(weights, 9, 3);
        fc.weights = weights;
        fc.biases = [1, 2, 3];

        output = forward_pass!(fc, input);
        expected = [270, 712, 1154];
        expected = reshape(expected, 3, 1);
        expected = convert(Array{Float32, 2}, expected);
        
        @test expected == output;
    end

    @testset "first layer example 2 -> 3" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8, 2, 8, 6, 4, 10, 7];
        input = reshape(input, 5, 3);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(5, 2);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        weights = reshape(weights, 5, 2);
        fc.weights = weights;
        fc.biases = [1, 2];

        output = forward_pass!(fc, input);
        expected = [88, 214, 73, 204, 108, 284];
        expected = reshape(expected, 2, 3);
        expected = convert(Array{Float32, 2}, expected);
        
        @test expected == output;
    end
end

function FCLayerTest_BackwardTest() 
    @testset begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 9, 1);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(9, 2);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        weights = reshape(weights, 9, 2);
        fc.weights = weights;
        fc.biases = [1, 2];

        output = forward_pass!(fc, input);

        y = [1, 0];
        y = reshape(y, 2, 1);
        y = convert(Array{Float32, 2}, y);

        loss, acc, grad = loss_and_accuracy(output, y);
        
        expected_dL_dX = [9, 9, 9, 9, 9, 9, 9, 9, 9];
        expected_dL_dX = reshape(expected_dL_dX, 9, 1);
        expected_dL_dX = convert(Array{Float32, 2}, expected_dL_dX);

        expected_dL_dW = [-1, -7, -5, -3, -9, -6, -6, -4, -8, 1, 7, 5, 3, 9, 6, 6, 4, 8];
        expected_dL_dW = reshape(expected_dL_dW, 9, 2);
        expected_dL_dW = convert(Array{Float32, 2}, expected_dL_dW);

        expected_dL_dB = [-1, 1];
        expected_dL_dB = convert(Array{Float32, 1}, expected_dL_dB);

        dL_dX = backward_pass!(fc, grad, input);

        @test expected_dL_dX == dL_dX;
        @test expected_dL_dW == fc.weights_gradient;
        @test expected_dL_dB == fc.biases_gradient;
    end

    @testset begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 9, 1);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(9, 3);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        weights = reshape(weights, 9, 3);
        fc.weights = weights;
        fc.biases = [1, 2, 3];

        output = forward_pass!(fc, input);

        y = [1, 0, 0];
        y = reshape(y, 3, 1);
        y = convert(Array{Float32, 2}, y);

        loss, acc, grad = loss_and_accuracy(output, y);
        
        expected_dL_dX = [18 18 18 18 18 18 18 18 18];
        expected_dL_dX = reshape(expected_dL_dX, 9, 1);
        expected_dL_dX = convert(Array{Float32, 2}, expected_dL_dX);

        expected_dL_dW = [-1, -7, -5, -3, -9, -6, -6, -4, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 5, 3, 9, 6, 6, 4, 8];
        expected_dL_dW = reshape(expected_dL_dW, 9, 3);
        expected_dL_dW = convert(Array{Float32, 2}, expected_dL_dW);

        expected_dL_dB = [-1, 0, 1];
        expected_dL_dB = convert(Array{Float32, 1}, expected_dL_dB);

        dL_dX = backward_pass!(fc, grad, input);

        @test expected_dL_dX == dL_dX;
        @test expected_dL_dW == fc.weights_gradient;
        @test expected_dL_dB == fc.biases_gradient;
    end

    @testset begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8, 2, 8, 6, 4, 10, 7];
        input = reshape(input, 5, 3);
        input = convert(Array{Float32, 2}, input);

        fc = FCLayer(5, 2);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        weights = reshape(weights, 5, 2);
        fc.weights = weights;
        fc.biases = [1, 2];

        output = forward_pass!(fc, input);

        y = [1, 0, 1, 0, 1, 0];
        y = reshape(y, 2, 3);
        y = convert(Array{Float32, 2}, y);

        loss, acc, grad = loss_and_accuracy(output, y);

        expected_dL_dX = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
        expected_dL_dX = reshape(expected_dL_dX, 5, 3);
        expected_dL_dX = convert(Array{Float32, 2}, expected_dL_dX);

        expected_dL_dW = [-15, -19, -13, -21, -18, 15, 19, 13, 21, 18];
        expected_dL_dW = reshape(expected_dL_dW, 5, 2);
        expected_dL_dW = convert(Array{Float32, 2}, expected_dL_dW);

        expected_dL_dB = [-3, 3];
        expected_dL_dB = convert(Array{Float32, 1}, expected_dL_dB);

        dL_dX = backward_pass!(fc, grad, input);

        @test expected_dL_dX == dL_dX;
        @test expected_dL_dW == fc.weights_gradient;
        @test expected_dL_dB == fc.biases_gradient;
    end
end

function FCLayerTest_UpdateWeights()
    @testset "first layer example 1 -> 2" begin
        layer = FCLayer(9, 2);
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        weights = reshape(weights, 5, 2);
        weights = convert(Array{Float32, 2}, weights);
        layer.weights = weights;
        
        biases = [1, 2];
        biases = convert(Array{Float32, 1}, biases);
        layer.biases = biases;

        gradients = [1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
        gradients = reshape(gradients, 5, 2);
        gradients = convert(Array{Float32, 2}, gradients);
        layer.weights_gradient = gradients;

        biases = [-1, 1];
        biases = convert(Array{Float32, 1}, biases);
        layer.biases_gradient = biases;

        update_weights(layer, 0.1, 1);

        expected_weights = [0.9, 1.8, 2.7, 3.6, 4.5, 6.6, 7.7, 8.8, 9.9, 11.0];
        expected_weights = reshape(expected_weights, 5, 2);
        expected_weights = convert(Array{Float32, 2}, expected_weights);

        expected_biases = [1.1, 1.9];
        expected_biases = convert(Array{Float32, 1}, expected_biases);

        @test expected_weights == layer.weights;
        @test expected_biases == layer.biases;
    end
end