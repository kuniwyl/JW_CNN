struct PoolLayer <: Layer
    pool_size::Tuple{Int, Int}
end


function forward_pass(layer::PoolLayer, input::Array)
    pool_height, pool_width = layer.pool_size
    input_height, input_width, input_channels = size(input)

    # Calculate the dimensions of the output feature maps
    # Ensuring that stride of 1 works with 2x2 pooling reducing the dimensions by half
    output_height = div(input_height, pool_height)
    output_width = div(input_width, pool_width)

    # Initialize the output array with very small numbers
    output = zeros((output_height, output_width, input_channels))

    # Store indices for the backward pass
    indices = fill((0, 0), (output_height, output_width, input_channels))

    for c in 1:input_channels
        for y in 1:output_height
            for x in 1:output_width
                # Extract the window
                window = input[(y-1)*pool_height+1:y*pool_height, (x-1)*pool_width+1:x*pool_width, c]
                max_value = maximum(window)
                output[y, x, c] = max_value

                # Capture the index of the max element for the backward pass
                local_idx = argmax(window)
                global_idx = ((y-1)*pool_height + div(local_idx[1], pool_height) + 1, (x-1)*pool_width + div(local_idx[2], pool_width) + 1)
                indices[y, x, c] = global_idx
            end
        end
    end

    return output, indices
end



function backward_pass(layer::PoolLayer, dL_dOut::Array, input::Array, indices::Array)
    input_height, input_width, input_channels = size(input)
    output_height, output_width, _ = size(dL_dOut)

    # Initialize gradient input as zeros
    dL_dIn = zeros(size(input))

    for c in 1:input_channels
        for i in 1:output_height
            for j in 1:output_width
                (max_y, max_x) = indices[i, j, c]
                dL_dIn[max_y, max_x, c] += dL_dOut[i, j, c]
            end
        end
    end

    return dL_dIn
end
