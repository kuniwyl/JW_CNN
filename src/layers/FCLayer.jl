mutable struct FCLayer <: Layer
    weights::Array{Float32, 2}  # Weights matrix of dimensions [input_dim, output_dim]
    biases::Array{Float32, 1}   # Biases vector of dimensions [output_dim]
    
    input::Array{Float32, 2}  # Input to the layer
    output::Array{Float32, 2}  # Output of the layer

    activation::Function  # Activation function
    deactivation::Function  # Derivative of the activation function

    weights_gradient::Array{Float32, 2}  # Gradient of the loss with respect to the weights
    biases_gradient::Array{Float32, 1}  # Gradient of the loss with respect to the biases

    dL_dX::Array{Float32, 2}  # Gradient of the loss with respect to the input
    dL_dZ::Array{Float32, 2}  # Gradient of the loss with respect to the output
end

# Function to initialize a new dense layer
function FCLayer(input_dim::Int, output_dim::Int, activation::Function=relu, deactivation::Function=relu_derivative)
    weights = 0.01 * randn(input_dim, output_dim)  # small random numbers
    biases = zeros(output_dim)
    
    input = zeros(1, 1)
    output = zeros(1, 1)

    weights_gradient = zeros(size(weights))
    biases_gradient = zeros(size(biases))
    
    dL_dX = zeros(1, 1)
    dL_dZ = zeros(1, 1)

    return FCLayer(weights, biases, input, output, activation, deactivation, weights_gradient, biases_gradient, dL_dX, dL_dZ)
end

function forward_pass(layer::FCLayer, input::Array)
    layer.input = input
    if size(layer.output) != (size(layer.weights, 2), size(input, 2))
        layer.output = zeros(size(layer.weights, 2), size(input, 2))
    end

    layer.output .= layer.weights' * input .+ layer.biases
    return layer.activation(layer.output)
end


function backward_pass(layer::FCLayer, dL_dY::Array)
    if size(layer.dL_dX) != size(layer.input)
        layer.dL_dX = zeros(size(layer.input))
        layer.dL_dZ = zeros(size(layer.output))
    end

    layer.dL_dZ .= layer.deactivation(layer.output)
    layer.dL_dZ .*= dL_dY

    # Calculate the gradients
    layer.dL_dX .= layer.weights * layer.dL_dZ
    layer.biases_gradient .+= sum(layer.dL_dZ, dims=2)
    layer.weights_gradient .+= layer.input * layer.dL_dZ'  

    return layer.dL_dX
end

function update_weights(layer::FCLayer, learning_rate::Float64, batch_size::Int64)
    layer.weights_gradient ./= batch_size
    layer.biases_gradient ./= batch_size
    layer.weights .-= learning_rate * layer.weights_gradient
    layer.biases .-= learning_rate * layer.biases_gradient

    # Reset the gradients
    layer.weights_gradient .= 0
    layer.biases_gradient .= 0
end