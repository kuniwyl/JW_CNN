function MaxPoolLaterTest_ForwardTest() 
    @testset "first layer example" begin
        input = [3, 3, 3, 3, 2, 1, 2, 4, 3, 3, 3, 3, 2, 1, 2, 5];
        input = reshape(input, 4, 4, 1, 1);
        input = convert(Array{Float32, 4}, input);

        pool = MaxPoolLayer(2, 2);
        output = forward_pass!(pool, input);

        expected = [3, 4, 3, 5];
        expected = reshape(expected, 2, 2, 1, 1);
        expected = convert(Array{Float32, 4}, expected);
        
        @test expected == output;
    end

    @testset "second layer example 2 -> 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 1);
        input = convert(Array{Float32, 4}, input);

        pool = MaxPoolLayer(2, 2);
        output = forward_pass!(pool, input);
        expected = [62, 86];
        expected = reshape(expected, 1, 1, 2, 1);
        expected = convert(Array{Float32, 4}, expected);
        expected_indices = [0, 1, 0, 0, 0, 0, 0, 1];
        expected_indices = reshape(expected_indices, 2, 2, 2, 1);
        expected_indices = convert(Array{Int64, 4}, expected_indices);

        @test expected == output;
        @test pool.indices == expected_indices;
    end

    @testset "third layer example batch_size 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 3);
        input = convert(Array{Float32, 4}, input);

        pool = MaxPoolLayer(2, 2);

        output = forward_pass!(pool, input)
        expected = [62, 86, 62, 86, 62, 86];
        expected = reshape(expected, 1, 1, 2, 3);
        expected = convert(Array{Float32, 4}, expected);
        expected_indices = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1];
        expected_indices = reshape(expected_indices, 2, 2, 2, 3);
        expected_indices = convert(Array{Int64, 4}, expected_indices);

        @test expected == output;
        @test pool.indices == expected_indices;
    end
end

function MaxPoolLayerTest_BackwardTest()
    @testset "second layer example 2 -> 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 1);
        input = convert(Array{Float32, 4}, input);

        pool = MaxPoolLayer(2, 2);
        output = forward_pass!(pool, input);
        expected = [62, 86];
        expected = reshape(expected, 1, 1, 2, 1);
        expected = convert(Array{Float32, 4}, expected);
        expected_indices = [0, 1, 0, 0, 0, 0, 0, 1];
        expected_indices = reshape(expected_indices, 2, 2, 2, 1);
        expected_indices = convert(Array{Int64, 4}, expected_indices);

        @test expected == output;

        dL_dY = [1, 2];
        dL_dY = reshape(dL_dY, 1, 1, 2, 1);
        dL_dY = convert(Array{Float32, 4}, dL_dY);
    
        dL_dX = backward_pass!(pool, dL_dY, input);
        expected = [0, 1, 0, 0, 0, 0, 0, 2];
        expected = reshape(expected, 2, 2, 2, 1);
        expected = convert(Array{Float32, 4}, expected);

        @test expected == dL_dX;
    end

    @testset "third layer example batch_size 3" begin
        input = [50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86, 50, 62, 57, 56, 59, 73, 54, 86];
        input = reshape(input, 2, 2, 2, 3);
        input = convert(Array{Float32, 4}, input);

        pool = MaxPoolLayer(2, 2);
        output = forward_pass!(pool, input);
        
        expected = [62, 86, 62, 86, 62, 86];
        expected = reshape(expected, 1, 1, 2, 3);
        expected = convert(Array{Float32, 4}, expected);

        expected_indices = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1];
        expected_indices = reshape(expected_indices, 2, 2, 2, 3);
        expected_indices = convert(Array{Int64, 4}, expected_indices);

        @test expected == output;

        dL_dY = [1, 2, 3, 4, 5, 6];
        dL_dY = reshape(dL_dY, 1, 1, 2, 3);
        dL_dY = convert(Array{Float32, 4}, dL_dY);

        dL_dX = backward_pass!(pool, dL_dY,  input);
        expected = [0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 6];
        expected = reshape(expected, 2, 2, 2, 3);
        expected = convert(Array{Float32, 4}, expected);

        @test expected == dL_dX;
    end
end