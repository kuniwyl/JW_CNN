struct FlattenLayer <: Layer
end

function forward_pass(layer::FlattenLayer, input::Array)
    return reshape(input, :)
end

function backward_pass(layer::FlattenLayer, gradient::Array, input::Array)
    # Simply reshape the gradient back to the original dimensions of the input
    return reshape(gradient, size(input))
end
