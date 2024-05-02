function pool_layer_test()
    @testset "Max Poll Layer Forward Tests" begin
        # Test the poll layer
        poll_layer = PoolLayer((2, 2))
        x = ones(25, 25, 6)
        x[1, 1, 1] = 10
        y = forward_pass(poll_layer, x)

        @test size(y[1]) == (12, 12, 6)  # Check output dimensions
        @test y[1][1, 1, 1] == 10  # Check the first element of the output

        poll_layer = PoolLayer((2, 2))
        y = forward_pass(poll_layer, y[1])
        @test size(y[1]) == (6, 6, 6)  # Check output dimensions
        @test y[1][1, 1, 1] == 10  # Check the first element of the output
    
    end

    @testset "Max Poll Layer Backward Tests" begin
        # Test the poll layer
        poll_layer = PoolLayer((2, 2))
        input = ones(4, 4, 1);
        input[:, :, 1] = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0; 13.0 14.0 15.0 16.0]
        gradient = ones(2, 2, 1)
        gradient[:, :, 1] = [1.0 2.0; 3.0 4.0]
        y, ids = forward_pass(poll_layer, input)
        @test isapprox(y, [6.0 8.0; 14.0 16.0], atol = 1e-3)  # Check the output
        # do not know how to test ids

        gradients = backward_pass(poll_layer, gradient, input, ids)
        @test isapprox(gradients, [0.0 0.0 0.0 0.0; 0.0 1.0 0.0 2.0; 0.0 0.0 0.0 0.0; 0.0 3.0 0.0 4.0], atol = 1e-3)  # Check the gradients
        # this test check ids - better would be to have test there 
    end

end