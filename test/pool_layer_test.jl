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
end