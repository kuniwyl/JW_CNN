mutable struct FlattenLayer <: Layer
    input_shape::Tuple
end

# Function to initialize a new flatten layer
function FlattenLayer()
    input_shape = (1, 1, 1, 1)
    return FlattenLayer(input_shape)
end

function forward_pass(layer::FlattenLayer, input::Array)
    layer.input_shape = size(input)

    return reshape(input, layer.input_shape[1] * layer.input_shape[2] * layer.input_shape[3], layer.input_shape[4])
end

# gradient from the next layer and the input from the previous layer
function backward_pass(layer::FlattenLayer, dL_dY::Array)
    return reshape(dL_dY, layer.input_shape)
end

# Update the weights of the network
function update_weights(layer::FlattenLayer, learning_rate::Float64, batch_size::Int64)
    
end