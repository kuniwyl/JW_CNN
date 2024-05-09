using Random

Random.seed!(1)

batch_size = 2
input = rand(1:2, 8, 8, 1, batch_size)
filters1 = rand(1:2, 3, 3, 2)

result = zeros(6, 6, 2, batch_size);
for x in 1:2
    filter = reshape(filters1[:, :, x], 3, 3, 1)
    for i in 1:6
        for j in 1:6
            r = input[i:i+2, j:j+2, :, :] .* filter
            result[i, j, x, :] += vec(sum(r, dims=(1, 2, 3)));
        end
    end
end
@show result

filters2 = rand(1:2, 3, 3, 3)
result1 = zeros(4, 4, 3, batch_size);
for x in 1:3
    filter = reshape(filters2[:, :, x], 3, 3, 1)
    for i in 1:4
        for j in 1:4
            r = result[i:i+2, j:j+2, :, :] .* filter
            result1[i, j, x, :] += vec(sum(r, dims=(1, 2, 3)))
        end
    end
end
@show result1

result2 = zeros(2, 2, 3, batch_size);
for x in 1:3
    for i in 1:2
        for j in 1:2
            r = result1[(i-1)*2+1:(i-1)*2+2, (j-1)*2+1:(j-1)*2+2, x, :]
            result2[i, j, x, :] = maximum(r, dims=(1, 2))
        end
    end
end
@show result2

result3 = zeros(12, batch_size);
for x in 1:batch_size
    result3[:, x] = vec(result2[:, :, :, x])
end
@show result3

result4 = zeros(10, batch_size)
weights = rand(12, 10)
for x in 1:batch_size
    result4[:, x] = weights' * result3[:, x]
end
@show result4

function softmax(x)
    x = x .- maximum(x)
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end

result5 = zeros(10, batch_size)
for x in 1:batch_size
    result5[:, x] = softmax(result4[:, x])
end
@show result5

# backward prop
grad_result5 = rand(10, batch_size)
grad_result4 = zeros(12, batch_size)
for x in 1:batch_size
    grad_result4[:, x] = weights * grad_result5[:, x]
end
@show grad_result4

grad_result3 = zeros(2, 2, 3, batch_size)
for x in 1:batch_size
    grad_result3[:, :, :, x] = reshape(grad_result4[:, x], 2, 2, 3)
end
@show grad_result3


grad_result2 = zeros(4, 4, 3, batch_size)
for x in 1:3
    for i in 1:2
        for j in 1:2
            grad_result2[(i-1)*2+1:(i-1)*2+2, (j-1)*2+1:(j-1)*2+2, x, :] = grad_result3[i, j, x, :]
        end
    end
end
@show grad_result2

grad_result1 = zeros(6, 6, 2, batch_size)
for x in 1:3
    filter = reshape(filters2[:, :, x], 3, 3, 1)
    for i in 1:4
        for j in 1:4
            r = grad_result2[i:i+2, j:j+2, :, :] .* filter
            grad_result1[i, j, x, :] += vec(sum(r, dims=(1, 2, 3)))
        end
    end
end
@show grad_result1

grad_result = zeros(8, 8, 1, batch_size)
for x in 1:2
    filter = reshape(filters1[:, :, x], 3, 3, 1)
    for i in 1:6
        for j in 1:6
            r = grad_result1[i:i+2, j:j+2, :, :] .* filter
            grad_result[i, j, :, :] += vec(sum(r, dims=(1, 2, 3)))
        end
    end
end
@show grad_result

