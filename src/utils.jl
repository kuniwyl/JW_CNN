function get_batches(inputs, targets, batch_size)
    n = size(inputs, 2)
    indices = shuffle(1:n)
    return [(inputs[:, idxs], targets[:, idxs]) for idxs in Iterators.partition(indices, batch_size)]
end

# Dummy functions for loss and accuracy, to be replaced with actual implementations
function compute_loss_and_gradient(output, target)
    # Compute loss (e.g., cross-entropy)
    loss = sum((output - target).^2) / size(output, 2)  # Mean squared error as example
    gradient = 2 * (output - target) / size(output, 2)  # Gradient of MSE
    return loss, gradient
end

function compute_accuracy(output, target)
    # Compute accuracy (assuming output and target are class labels)
    predictions = argmax(output, dims=1)
    target_labels = argmax(target, dims=1)
    accuracy = mean(predictions .== target_labels)
    return accuracy
end
