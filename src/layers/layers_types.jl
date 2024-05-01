abstract type Layer end

forward_pass(layer::Layer, input::Array) = error("Not implemented")
backward_pass(layer::Layer, gradient::Array) = error("Not implemented")
update_weights!(layer::Layer, learning_rate::Float64) = error("Not implemented")
