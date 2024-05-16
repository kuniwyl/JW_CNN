mutable struct FlattenLayer <: Layer
    input_shape::Tuple
end

function FlattenLayer()
    input_shape = (1, 1, 1, 1)
    return FlattenLayer(input_shape)
end

function forward_pass(layer::FlattenLayer, input::Array)
    layer.input_shape = size(input)
    return reshape(input, :, layer.input_shape[end])
end

function backward_pass(layer::FlattenLayer, dL_dY::Array)
    return reshape(dL_dY, layer.input_shape)
end

function update_weights(layer::FlattenLayer, learning_rate::Float64, batch_size::Int64)
    # No weights to update in FlattenLayer
end