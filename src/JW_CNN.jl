module JW_CNN

using Statistics: mean
using LinearAlgebra
using Flux
using Random

# Utils
include("utils.jl")

# Layers
include("layers/Layer.jl")
include("layers/ConvLayer.jl")
include("layers/MaxPoolLayer.jl")
include("layers/FCLayer.jl")
include("layers/FlattenLayer.jl")

mutable struct NeuralNetwork
    layers::Vector  # A vector to hold different types of layers
    learning_rate::Float64
end

# Constructor for the neural network
function NeuralNetwork(learning_rate::Float64)
    return NeuralNetwork([], learning_rate)
end

# Function to add layers to the network
function add_layer!(network::NeuralNetwork, layer)
    push!(network.layers, layer)
end

# Forward pass through the network
function forward_pass(network::NeuralNetwork, input)
    output = input
    for layer in network.layers
        output = forward_pass(layer, output)
    end
    return output
end

# Backward pass through the network
function backward_pass(network::NeuralNetwork, dL_dY)
    dL_dOut = dL_dY
    for layer in reverse(network.layers)
        dL_dOut = backward_pass(layer, dL_dOut)
    end
end

# Update the weights of the network
function update_weights!(network::NeuralNetwork, batch_size)
    for layer in network.layers
        update_weights(layer, network.learning_rate, batch_size)
    end
end

function shuffle_data(inputs, targets)
    num_samples = size(targets, 2)

    # Create an array of indices and shuffle it
    indices = Random.shuffle(1:num_samples)

    # Shuffle inputs and targets based on the shuffled indices
    shuffled_inputs = inputs[:, :, :, indices]
    shuffled_targets = targets[:, indices]

    return shuffled_inputs, shuffled_targets
end

function get_batches(inputs, targets, batch_size)
    # Shuffle inputs and targets
    shuffled_inputs, shuffled_targets = shuffle_data(inputs, targets)

    num_samples = size(shuffled_targets, 2)
    num_batches = div(num_samples, batch_size)

    batches = []
    for i in 1:num_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = i * batch_size

        # Get the batches from the shuffled inputs and targets
        batch_indices = start_idx:end_idx
        push!(batches, (shuffled_inputs[:, :, :, batch_indices], shuffled_targets[:, batch_indices]))
    end
    return batches
end

function cross_entropy_loss(output, target)
    return -sum(target .* log.(output))
end

function calculate_accuracy(output, target)
    predictions = argmax(output, dims=1)
    target_labels = argmax(target, dims=1)
    return mean(predictions .== target_labels)
end

function train(network::NeuralNetwork, inputs, targets, epochs::Int64, batch_size::Int64)
    for epoch in 1:epochs
        @info "Epoch $epoch"
        total_loss = 0.0
        total_accuracy = 0.0
        batches = get_batches(inputs, targets, batch_size)
        
        for (x, y) in batches
            # Forward pass
            output = forward_pass(network, x)

            loss, acc, grad = loss_and_accuracy(output, y)
            
            # Backward pass
            backward_pass(network, grad)

            # Update weights
            update_weights!(network, batch_size)

            total_loss += loss
            total_accuracy += acc
        end
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / length(batches)
        avg_accuracy = total_accuracy / length(batches)
        @info "Average Loss: $avg_loss, Average Accuracy: $avg_accuracy"
    end
    return network.layers
end

function test(network::NeuralNetwork, test_inputs, test_targets)
    output = forward_pass(network, test_inputs)

    loss, acc, _ = loss_and_accuracy(output, test_targets)

    return loss, acc
end


export train!, NeuralNetwork, ConvLayer, MaxPoolLayer, FCLayer, FlattenLayer, forward_pass, backward_pass, update_weights!, update_weights, calculate_dl_dOut, loss_and_accuracy

end 
