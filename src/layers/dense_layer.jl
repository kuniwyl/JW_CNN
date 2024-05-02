struct DenseLayer
    weights::Array{Float64, 2}  # Weights matrix of dimensions [input_dim, output_dim]
    biases::Array{Float64, 1}   # Biases vector of dimensions [output_dim]
    activation::Function        # Activation function (e.g., relu)
end

# Function to initialize a new dense layer
function DenseLayer(input_dim::Int, output_dim::Int, activation::Function)
    weights = 0.01 * randn(input_dim, output_dim)  # small random numbers
    biases = zeros(output_dim)
    DenseLayer(weights, biases, activation)
end

function derivative_relu(z)
    return z .> 0 .* 1.0  # Broadcast comparison, returning 1.0 where true, 0.0 otherwise
end

function forward_pass(layer::DenseLayer, input::Array{Float64, 1})  # Single example forward pass
    # Calculate the pre-activation values (linear combination before ReLU)
    preactivation = layer.weights' * input + layer.biases

    # Apply the activation function
    output = layer.activation(preactivation)

    return output
end


function backward_pass(layer::DenseLayer, y::Array{Float64, 1}, input::Array{Float64, 1}, true_output::Array{Float64, 1})
    # Calculate the pre-activation values (linear combination before ReLU)
    preactivation = layer.weights' * input + layer.biases

    # Gradient of the loss with respect to the output
    dL_dOut = y - true_output

    # Calculate the derivative of ReLU
    dOut_dZ = derivative_relu(preactivation)

    # Gradient of the loss with respect to the pre-activation (chain rule application)
    dL_dZ = dL_dOut .* dOut_dZ

    # Gradient of the loss with respect to the weights
    dL_dW = input * dL_dZ'  # Outer product of dL_dZ and input

    # Gradient of the loss with respect to the biases
    dL_dB = dL_dZ

    return dL_dW, dL_dB
end


function update_weights!(layer::DenseLayer, dL_dW::Array, dL_dB::Array, learning_rate::Float64)
    layer.weights .-= learning_rate .* dL_dW
    layer.biases .-= learning_rate .* dL_dB
end
