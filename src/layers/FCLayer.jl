mutable struct FCLayer <: Layer
    weights::Array{Float32, 2}  # Weights matrix of dimensions [input_dim, output_dim]
    biases::Array{Float32, 1}   # Biases vector of dimensions [output_dim]
    
    input::Array{Float32, 2}  # Input to the layer
    output::Array{Float32, 2}  # Output of the layer

    activation::Function  # Activation function
    deactivation::Function  # Derivative of the activation function

    weights_gradient::Array{Float32, 2}  # Gradient of the loss with respect to the weights
    biases_gradient::Array{Float32, 1}  # Gradient of the loss with respect to the biases
end

function relu(z)
    return max.(0, z)
end

function derivative_relu(z)
    return z .> 0 .* 1.0  # Broadcast comparison, returning 1.0 where true, 0.0 otherwise
end

# Function to initialize a new dense layer
function FCLayer(input_dim::Int, output_dim::Int, activation::Function=relu, deactivation::Function=derivative_relu)
    weights = 0.01 * randn(input_dim, output_dim)  # small random numbers
    biases = zeros(output_dim)
    
    input = zeros(1, 1)
    output = zeros(1, 1)

    weights_gradient = zeros(size(weights))
    biases_gradient = zeros(size(biases))
    
    return FCLayer(weights, biases, input, output, activation, deactivation, weights_gradient, biases_gradient)
end

function forward_pass(layer::FCLayer, input::Array)      
    layer.input = input
    
    if size(layer.output) == (1, 1)
        layer.output = zeros(size(layer.weights, 2), size(input, 2))
    end

    for x in 1:size(input, 2)
        layer.output[:, x] = layer.weights' * input[:, x] .+ layer.biases
    end
    
    return layer.activation(layer.output)
end


function backward_pass(layer::FCLayer, dL_dY::Array)
    dL_dX = zeros(size(layer.input))

    # Calculate the derivative of ReLU
    dL_dZ = layer.deactivation(layer.output) .* dL_dY

    # Calculate the gradients
    layer.weights_gradient .+= layer.input * dL_dZ'

    layer.biases_gradient .+= sum(dL_dZ, dims=2)
    dL_dX = layer.weights * dL_dZ

    return dL_dX
end


function update_weights(layer::FCLayer, learning_rate::Float64, batch_size::Int64)
    layer.weights .-= learning_rate * layer.weights_gradient / batch_size
    layer.biases .-= learning_rate * layer.biases_gradient / batch_size

    # Reset the gradients
    layer.weights_gradient .= 0
    layer.biases_gradient .= 0
end