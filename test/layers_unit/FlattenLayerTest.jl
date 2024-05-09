function FlattenLayerTest_ForwardTest() 
    @testset "first layer example 1 -> 2" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 3, 3, 1, 1);

        flatten = FlattenLayer();
        output = forward_pass!(flatten, input);

        expected = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        expected = reshape(expected, 9, 1);
        expected = convert(Array{Float32, 2}, expected);
        
        @test expected == output;
    end

    @testset "second layer example 2 -> 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 1);

        flatten = FlattenLayer();
        output = forward_pass!(flatten, input);

        expected = [50, 62, 57, 56, 59, 73, 54, 86];
        expected = reshape(expected, 8, 1);
        expected = convert(Array{Float32, 2}, expected);

        @test expected == output;
    end

    @testset "third layer example batch_size 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 3);
        
        flatten = FlattenLayer();
        output = forward_pass!(flatten, input)
        expected = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        expected = reshape(expected, 8, 3);
        expected = convert(Array{Float32, 2}, expected);

        @test expected == output;
    end
end

function FlattenLayerTest_BackwardTest()
    @testset "first layer example 1 -> 2" begin
        input = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        input = reshape(input, 3, 3, 1, 1);
        input = convert(Array{Float32, 4}, input);

        flatten = FlattenLayer();
        output = forward_pass!(flatten, input);

        expected = [1, 7, 5, 3, 9, 6, 6, 4, 8];
        expected = reshape(expected, 9, 1);
        expected = convert(Array{Float32, 2}, expected);
        
        @test expected == output;

        output = output .+ 1;
        dL_dOut = output;

        dL_dX = backward_pass!(flatten, dL_dOut, input);
        expected_dL_dX = [2, 8, 6, 4, 10, 7, 7, 5, 9];
        expected_dL_dX = reshape(expected_dL_dX, 3, 3, 1, 1);
        expected_dL_dX = convert(Array{Float32, 4}, expected_dL_dX);

        @test expected_dL_dX == dL_dX;
    end

    @testset "second layer example 2 -> 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 1);
        input = convert(Array{Float32, 4}, input);

        flatten = FlattenLayer();
        output = forward_pass!(flatten, input);

        expected = [50, 62, 57, 56, 59, 73, 54, 86];
        expected = reshape(expected, 8, 1);
        expected = convert(Array{Float32, 2}, expected);

        @test expected == output;

        output = output .+ 1;
        dL_dOut = output;

        dL_dX = backward_pass!(flatten, dL_dOut, input);
        expected_dL_dX = [51, 63, 58, 57, 60, 74, 55, 87];
        expected_dL_dX = reshape(expected_dL_dX, 2, 2, 2, 1);
        expected_dL_dX = convert(Array{Float32, 4}, expected_dL_dX);

        @test expected_dL_dX == dL_dX;
    end

    @testset "third layer example batch_size 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 3);
        input = convert(Array{Float32, 4}, input);

        flatten = FlattenLayer();
        output = forward_pass!(flatten, input);
        expected = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        expected = reshape(expected, 8, 3);
        expected = convert(Array{Float32, 2}, expected);

        @test expected == output;

        output = output .+ 1;
        dL_dOut = output;

        dL_dX = backward_pass!(flatten, dL_dOut, input);
        expected_dL_dX = [51, 63, 58, 57, 60, 74, 55, 87, 51, 63, 58, 57, 60, 74, 55, 87, 51, 63, 58, 57, 60, 74, 55, 87];
        expected_dL_dX = reshape(expected_dL_dX, 2, 2, 2, 3);
        expected_dL_dX = convert(Array{Float32, 4}, expected_dL_dX);

        @test expected_dL_dX == dL_dX;
    end
end