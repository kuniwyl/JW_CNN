module JW_CNN

# Utils
include("utils.jl")

# Layers
include("layers/layers_types.jl")
include("layers/conv_layer.jl")
include("layers/pool_layer.jl")
include("layers/dense_layer.jl")
include("layers/flaten_layer.jl")

struct NeuralNetwork
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


function train(network::NeuralNetwork, inputs, targets, epochs, batch_size)
    for epoch in 1:epochs
        total_loss = 0.0
        total_accuracy = 0.0
        batches = get_batches(inputs, targets, batch_size)
        
        for (batch_inputs, batch_targets) in batches
            # Forward pass
            output = forward_pass(network, batch_inputs)
            
            # Compute loss and gradient
            loss, output_gradient = compute_loss_and_gradient(output, batch_targets)
            total_loss += loss
            
            # Compute accuracy if needed
            if size(output, 1) == size(batch_targets, 1)  # Assuming one-hot or class labels alignment
                accuracy = compute_accuracy(output, batch_targets)
                total_accuracy += accuracy
            end
            
            # Backward pass
            gradients = backward_pass(network, output_gradient, batch_inputs)
            
            # Update weights
            update_weights!(network, gradients)
        end
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / length(batches)
        avg_accuracy = total_accuracy / length(batches)
        
        println("Epoch $epoch: Loss = $avg_loss, Accuracy = $avg_accuracy")
    end
end

export train, NeuralNetwork, ConvLayer, PoolLayer, DenseLayer, FlattenLayer, forward_pass, backward_pass, update_weights!, calculate_dl_dOut

end
