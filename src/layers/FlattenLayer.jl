mutable struct FlattenLayer <: Layer
    input_shape::Tuple

    output::Array{Float32, 2}
end

# Function to initialize a new flatten layer
function FlattenLayer()
    input_shape = (1, 1, 1, 1)
    output = zeros(1, 1)
    return FlattenLayer(input_shape, output)
end

function forward_pass(layer::FlattenLayer, input::Array)
    if layer.output == zeros(1, 1)
        layer.input_shape = size(input)
        input_height, input_width, input_channels, batch_size = size(input)
        layer.output = zeros(input_height * input_width * input_channels, batch_size)
    else
        layer.output .= 0
    end

    return reshape(input, size(layer.output))
end

# gradient from the next layer and the input from the previous layer
function backward_pass(layer::FlattenLayer, dL_dY::Array)
    return reshape(dL_dY, layer.input_shape)
end

# Update the weights of the network
function update_weights(layer::FlattenLayer, learning_rate::Float64, batch_size::Int64)
    
end