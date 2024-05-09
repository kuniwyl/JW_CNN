using Random
using LinearAlgebra

function bsize_case_full_test_gen(batch_size::Int=3)
    Random.seed!(1)

    # generate random float matrix 8x8
    matrix = rand(8, 8, 1, batch_size)

    #create filter 2x3x3
    filters1 = rand(3, 3, 2)
    # apply filter to matrix
    result = zeros(6, 6, 2, batch_size)
    for x in 1:2
        filter = filters1[:, :, x]
        for i in 1:6
            for j in 1:6
                r = matrix[i:i+2, j:j+2, :, :] .* filter
                result[i, j, x, :] .+= sum(reshape(r, :, batch_size), dims=1)
            end
        end
    end

    filters2 = rand(3, 3, 3)
    result1 = zeros(4, 4, 3, batch_size)
    for x in 1:3
        filter = filters2[:, :, x]
        for i in 1:4
            for j in 1:4
                r = result[i:i+2, j:j+2, :, :] .* filter
                @show r
                result1[i, j, x, :] += squeeze(sum(r, dims=(1, 2, 3)), dims=(1, 2, 3))
            end
        end
    end

    return result, result1, filters1, filters2, matrix

    # max pooling
    result2 = zeros(batch_size, 3, 2, 2)
    for x in 1:3
        for i in 1:2
            for j in 1:2
                r = result1[:, x, (i-1)*2+1:(i-1)*2+2, (j-1)*2+1:(j-1)*2+2]
                result2[:, x, i, j] = maximum(r, dims=(2, 3))
            end
        end
    end

    # flatten
    result3 = zeros(batch_size, 12)
    for x in 1:batch_size
        result3[x, :] = vec(result2[x, :, :, :])
    end

    # fully connected layer
    weights = rand(12, 10)
    result4 = result3 * weights

    # softmax
    function softmax(x)
        exp_x = exp.(x)
        return exp_x ./ sum(exp_x)
    end

    result5 = zeros(batch_size, 10)
    for x in 1:batch_size
        result5[x, :] = softmax(result4[x, :])
    end

    return result, result1, result2, result3, result4, result5, weights, filters1, filters2, matrix
end

result, result1, result2, result3, result4, result5, weights, filters1, filters2, matrix = bsize_case_full_test_gen(3)