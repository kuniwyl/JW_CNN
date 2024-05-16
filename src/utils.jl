using Statistics

function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=1))
    return exp_x ./ sum(exp_x, dims=1)
end

function cross_entropy_loss_with_gradient(predictions, targets)
    probabilities = softmax(predictions)
    loss = -mean(sum(targets .* log.(probabilities), dims=1))
    gradient = probabilities - targets
    return loss, Float32.(gradient)
end

function one_cold(encoded)
    return [argmax(vec) for vec in eachcol(encoded)]
end

function loss_and_accuracy(ŷ, y)
    loss, grad = cross_entropy_loss_with_gradient(ŷ, y)
    pred_classes = one_cold(ŷ)
    true_classes = one_cold(y)
    acc = round(100 * mean(pred_classes .== true_classes); digits=2)
    return loss, acc, grad
end

function shuffle_data(inputs, targets)
    num_samples = size(inputs, 4)
    indices = Random.shuffle(1:num_samples)
    shuffled_inputs = inputs[:, :, :, indices]
    shuffled_targets = targets[:, indices]
    return shuffled_inputs, shuffled_targets
end

function get_batches(inputs, targets, batch_size)
    shuffled_inputs, shuffled_targets = shuffle_data(inputs, targets)
    num_samples = size(shuffled_targets, 2)
    num_batches = div(num_samples, batch_size)
    batches = []
    for i in 1:num_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = i * batch_size
        batch_indices = start_idx:end_idx
        push!(batches, (shuffled_inputs[:, :, :, batch_indices], shuffled_targets[:, batch_indices]))
    end
    return batches
end