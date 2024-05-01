struct ConvLayer <: Layer
    weights::Array{Float32, 4}  # (filter_height, filter_width, input_channels, output_channels)
    biases::Array{Float32, 1}
    stride::Int
    padding::Int
end


function forward_pass(layer::ConvLayer, input::Array)
    # Extracting dimensions from the input and layer properties
    input_height, input_width, input_channels = size(input)
    filter_height, filter_width, _, num_filters = size(layer.weights)
    stride = layer.stride
    padding = layer.padding

    # Calculate output dimensions
    output_height = div(input_height + 2 * padding - filter_height, stride) + 1
    output_width = div(input_width + 2 * padding - filter_width, stride) + 1

    # Initialize the output array with zeros
    output = zeros(output_height, output_width, num_filters)

    # Apply padding to the input
    padded_input = zeros(input_height + 2 * padding, input_width + 2 * padding, input_channels)
    padded_input[padding+1:end-padding, padding+1:end-padding, :] .= input

    # Convolution operation
    for f in 1:num_filters
        for y in 1:stride:(input_height - filter_height + 1 + 2*padding)
            for x in 1:stride:(input_width - filter_width + 1 + 2*padding)
                for c in 1:input_channels
                    # Compute the receptive field
                    receptive_field = padded_input[y:y+filter_height-1, x:x+filter_width-1, c]
                    # Convolution: element-wise multiplication and sum
                    output[div(y-1, stride)+1, div(x-1, stride)+1, f] +=
                        sum(receptive_field .* layer.weights[:, :, c, f])
                end
                # Add bias after summing over all channels
                output[div(y-1, stride)+1, div(x-1, stride)+1, f] += layer.biases[f]
            end
        end
    end

    # Activation Function (ReLU)
    return max.(0, output)
end


function backward_pass(layer::ConvLayer, dL_dOut::Array, input::Array)
    filter_height, filter_width, _, num_filters = size(layer.weights)
    input_height, input_width, input_channels = size(input)
    stride = layer.stride
    padding = layer.padding

    # Initialize gradients with respect to weights, biases, and input
    dL_dW = zeros(size(layer.weights))
    dL_dB = zeros(size(layer.biases))
    dL_dIn = zeros(size(input))

    # Pad the input and initialize padded input gradient
    padded_input = zeros(input_height + 2 * padding, input_width + 2 * padding, input_channels)
    padded_input[padding+1:end-padding, padding+1:end-padding, :] .= input
    padded_dL_dIn = zeros(size(padded_input))

    # Iterate over each filter
    for f in 1:num_filters
        for y in 1:stride:(input_height - filter_height + 1 + 2*padding)
            for x in 1:stride:(input_width - filter_width + 1 + 2*padding)
                y_out = div(y-1, stride)+1
                x_out = div(x-1, stride)+1
                # Compute gradients for each part
                for c in 1:input_channels
                    # Extract the current slice of the input
                    receptive_field = padded_input[y:y+filter_height-1, x:x+filter_width-1, c]

                    # Gradient with respect to the output (dL/dOut) multiplied by the receptive field
                    dL_dW[:, :, c, f] += dL_dOut[y_out, x_out, f] * receptive_field

                    # Gradient with respect to the input
                    padded_dL_dIn[y:y+filter_height-1, x:x+filter_width-1, c] +=
                        dL_dOut[y_out, x_out, f] * layer.weights[:, :, c, f]
                end
                # Gradient with respect to the biases
                dL_dB[f] += dL_dOut[y_out, x_out, f]
            end
        end
    end

    # Remove padding from the gradient with respect to the input
    dL_dIn .= padded_dL_dIn[padding+1:end-padding, padding+1:end-padding, :]

    return dL_dIn, dL_dW, dL_dB
end


function update_weights!(layer::ConvLayer, dL_dW::Array, dL_dB::Array, learning_rate::Float64)
    # Update the weights by subtracting a fraction of the gradient
    layer.weights .-= learning_rate .* dL_dW

    # Update the biases in a similar manner
    layer.biases .-= learning_rate .* dL_dB
end

