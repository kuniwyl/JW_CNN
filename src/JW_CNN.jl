module JW_CNN

using Statistics: mean
using LinearAlgebra
using Flux
using Random
using Base.Threads: nthreads, @threads
using DataStructures

println("JW_CNN module loaded")
println("Number of threads: ", nthreads())

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
    batch_size::Int64
end

# Constructor for the neural network
function NeuralNetwork(learning_rate::Float64, batch_size::Int64)
    return NeuralNetwork([], learning_rate, batch_size)
end

# Function to add layers to the network
function add_layer!(network::NeuralNetwork, layer)
    push!(network.layers, layer)
end

# Forward pass through the network
function forward_pass(network::NeuralNetwork, input)
    output = input
    # stack = Stack{Array}()
    for layer in network.layers
        # push!(stack, output) # Save the input for the backward pass
        output = forward_pass(layer, output)    
    end
    # push!(stack, output) # Save the output of the network
    return output
end

# Backward pass through the network
function backward_pass(network::NeuralNetwork, dL_dY, stack)
    dL_dOut = dL_dY
    for layer in reverse(network.layers)
        # input = pop!(stack) # Get the input from the forward pass
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
    num_samples = size(inputs, 4)

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

function train(network::NeuralNetwork, inputs, targets, test_input, test_target, epochs::Int64)
    for epoch in 1:epochs
        @time begin
        @info "Epoch $epoch"
        total_loss = 0.0
        total_accuracy = 0.0
        batches = get_batches(inputs, targets, network.batch_size)
        i = 1
        
        for (x, y) in batches
            # Forward pass
            # stack = forward_pass(network, x)
            # output = pop!(stack) # Get the output of the network
            output = forward_pass(network, x)

            loss, acc, grad = loss_and_accuracy(output, y)
            
            # Backward pass
            backward_pass(network, grad, stack)

            # Update weights
            update_weights!(network, network.batch_size)

            total_loss += loss
            total_accuracy += acc

            if i % 50 == 0
                @info "Loss: $loss, Accuracy: $acc, Epoch $epoch, Batch number: $i / $(length(batches))"
            end
            i += 1
        end
        
        test_loss, test_acc = test(network, test_input, test_target)
        @info "Test Loss: $test_loss, Test Accuracy: $test_acc"
        end
    end
    return network.layers
end



function test(network::NeuralNetwork, test_inputs, test_targets)
    total_loss = 0.0
    total_accuracy = 0.0
    batches = get_batches(test_inputs, test_targets, network.batch_size)

    for (x, y) in batches
        output = forward_pass(network, x)
        # loss, acc, _ = loss_and_accuracy(pop!(output), y)
        loss, acc, _ = loss_and_accuracy(output, y)
        total_loss += loss
        total_accuracy += acc
    end

    total_loss /= length(batches)
    total_accuracy /= length(batches)
    @info "Test Loss: $total_loss, Test Accuracy: $total_accuracy"
    return total_loss, total_accuracy
end


export train!, NeuralNetwork, ConvLayer, MaxPoolLayer, FCLayer, FlattenLayer, forward_pass, backward_pass, forward_pass!, backward_pass!, update_weights!, update_weights, calculate_dl_dOut, loss_and_accuracy

end 
