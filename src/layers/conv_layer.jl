struct ConvLayer <: Layer
    weights::Array{Float32, 4}  # (filter_height, filter_width, input_channels, output_channels)
    biases::Array{Float32, 1}
end


function forward_pass(layer::ConvLayer, input::Array)
    # Extracting dimensions from the input and layer properties
    input_height, input_width, input_channels = size(input)
    filter_height, filter_width, _, num_filters = size(layer.weights)

    # Calculate output dimensions
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    # Initialize the output array with zeros
    output = zeros(output_height, output_width, num_filters)

    # Convolution operation
    for f in 1:num_filters
        for y in 1:output_height
            for x in 1:output_width
                receptive_field = input[y:y+filter_height-1, x:x+filter_width-1, :]
                output[y, x, f] = sum(receptive_field .* layer.weights[:, :, :, f]) + layer.biases[f]
            end
        end
    end

    # Activation Function (ReLU)
    return max.(0, output)
end


function backward_pass(layer::ConvLayer, dL_dOut::Array, input::Array)
    filter_height, filter_width, _, num_filters = size(layer.weights)
    input_height, input_width, input_channels = size(input)

    # Initialize gradients with respect to weights, biases, and input
    dL_dW = zeros(size(layer.weights))
    dL_dB = zeros(size(layer.biases))
    dL_dIn = zeros(size(input))

    # Iterate over each filter
    for f in 1:num_filters
        for y in 1:(input_height - filter_height + 1)
            for x in 1:(input_width - filter_width + 1)
                y_out = y
                x_out = x
                # Compute gradients for each part
                for c in 1:input_channels
                    # Extract the current slice of the input
                    receptive_field = input[y:y+filter_height-1, x:x+filter_width-1, c]

                    # Gradient with respect to the output (dL/dOut) multiplied by the receptive field
                    dL_dW[:, :, c, f] += receptive_field .* dL_dOut[y_out, x_out, f]

                    # Gradient with respect to the input
                    dL_dIn[y:y+filter_height-1, x:x+filter_width-1, c] += layer.weights[:, :, c, f] .* dL_dOut[y_out, x_out, f]
                end
                # Gradient with respect to the biases
                dL_dB[f] += dL_dOut[y_out, x_out, f]
            end
        end
    end

    return dL_dW, dL_dB
end


function update_weights!(layer::ConvLayer, dL_dW::Array, dL_dB::Array, learning_rate::Float64)
    # Update the weights by subtracting a fraction of the gradient
    layer.weights .-= learning_rate .* dL_dW

    # Update the biases in a similar manner
    layer.biases .-= learning_rate .* dL_dB
end

