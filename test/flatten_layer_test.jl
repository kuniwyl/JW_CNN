function flatten_layer_test()
    @testset "Flatten Layer Forward Tests" begin
        # Test the flatten layer
        input = rand(3, 4, 5)
        layer = FlattenLayer()
        output = forward_pass(layer, input)
        @test size(output) == (60,)

        input = rand(2, 3, 4, 5)
        layer = FlattenLayer()
        output = forward_pass(layer, input)
        @test size(output) == (120,)
    end

    @testset "Flatten Layer Backward Tests" begin
        # Test the flatten layer
        input = rand(3, 4, 5)
        layer = FlattenLayer()
        output = forward_pass(layer, input)
        gradients = backward_pass(layer, output, input)
        @test size(gradients) == (3, 4, 5)

        input = rand(2, 3, 4, 5)
        layer = FlattenLayer()
        output = forward_pass(layer, input)
        gradients = backward_pass(layer, output, input)
        @test size(gradients) == (2, 3, 4, 5)
    end
end