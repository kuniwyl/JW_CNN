function dense_layer_test()
# Test the DenseLayer functionality
@testset "DenseLayer Tests" begin
    @testset "Forward Pass Test" begin
        input = ones(10)
        layer = DenseLayer(ones(10, 5), ones(5), relu)
        output, preactivation = forward_pass(layer, input)
        @test size(output) == (5,)  #Check output dimensions
        @test output[1] == 11  # Check the first element of the output

        layer = DenseLayer(ones(5, 2), ones(2), relu)
        output, pre = forward_pass(layer, output)
        @test size(output) == (2,)  # Check output dimensions
        @test output[1] == 56  # Check the first element of the output
    end

    @testset "Backward Pass Test" begin
        inputs = vec([1.0 2.0 3.0]);
        true_output = vec([1.0 0.0]);
        w1 = [1.76405235 0.40015721; 0.97873798 2.2408932; 1.86755799 -0.97727788]
        w1 = reshape(w1, 3, 2);
        b1 = [0.95008842 -0.15135721];
        b1 = reshape(b1, :);
        w2 = [-0.10321885  0.4105985; 0.14404357  1.45427351];
        w2 = reshape(w2, 2, 2);
        b2 = [ 0.76103773  0.12167502];
        b2 = reshape(b2, :);
        learning_rate = 0.01;

        layer1 = DenseLayer(w1, b1, relu);
        layer2 = DenseLayer(w2, b2, relu);

        a1, z1 = forward_pass(layer1, vec(inputs));
        output, z2 = forward_pass(layer2, a1);
        @test isapprox(a1, vec([10.2742907 1.79875276]), atol = 1e-3)
        @test isapprox(z1, vec([10.2742907 1.79875276]), atol = 1e-3)
        @test isapprox(output, vec([0.0 6.95616187]), atol = 1e-3)
        @test isapprox(z2, vec([-0.04036399 6.95616187]), atol = 1e-3)

        loss = 0.5 * sum((output - true_output).^2);
        @test loss ≈ 24.694093981092983
        
        dl_dout = calculate_dl_dOut(output, true_output);
        dl_dW2, dl_dB2 = backward_pass(layer2, dl_dout, z2, a1);
        @test isapprox(dl_dW2, [-0.0 71.46962922; -0.0 12.51241535], atol = 1e-3)
        @test isapprox(dl_dB2, vec([-0.0 6.95616187]), atol = 1e-3)

        dl_da1 = calculate_dl_dOut(layer2.weights, dl_dB2);
        dl_dW1, dl_dB1 = backward_pass(layer1, dl_da1, z1, inputs);
        @test isapprox(dl_dW1, [2.85618964 10.11616192; 5.71237929 20.23232384; 8.56856893 30.34848575], atol = 1e-3)
        @test isapprox(dl_dB1, vec([2.85618964 10.11616192]), atol = 1e-3)

        update_weights!(layer1, dl_dW1, dl_dB1, learning_rate);
        update_weights!(layer2, dl_dW2, dl_dB2, learning_rate);
        @test isapprox(layer1.weights, [1.73549045 0.29899559; 0.92161419 2.03856996; 1.7818723 -1.28076274], atol = 1e-3)
        @test isapprox(layer1.biases, vec([0.92152652 -0.25251883]), atol = 1e-3)
        @test isapprox(layer2.weights, [-0.10321885 -0.30409779; 0.14404357  1.32914935], atol = 1e-3)
        @test isapprox(layer2.biases, vec([0.76103773 0.0521134]), atol = 1e-3)
    end
end
end

function relu(z)
    return max.(0, z)
end

function relu_derivative(z)
    return z .> 0
end