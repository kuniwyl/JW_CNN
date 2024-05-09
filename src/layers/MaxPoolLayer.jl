mutable struct MaxPoolLayer <: Layer
    pool_size::Tuple{Int, Int}
    
    input::Array{Float32, 4}
    output::Array{Float32, 4}
    indices::Array{Float32, 4}
end

function MaxPoolLayer(height::Int, width::Int)::MaxPoolLayer
    input = zeros(1, 1, 1, 1)
    output = zeros(1, 1, 1, 1)
    indices = zeros(1, 1, 1, 1)
    
    return MaxPoolLayer((height, width), input, output, indices)
end

function forward_pass(layer::MaxPoolLayer, input::Array)
    layer.input = input
    pool_height, pool_width = layer.pool_size
    input_height, input_width, input_channels, batch_size = size(input)

    output_height = div(input_height, pool_height)
    output_width = div(input_width, pool_width)

    if size(layer.output) == (1, 1, 1, 1)
        layer.output = zeros(output_height, output_width, input_channels, batch_size)
    else
        layer.output .= 0
    end

    if size(layer.indices) == (1, 1, 1, 1)
        layer.indices = zeros(input_height, input_width, input_channels, batch_size)
    else
        layer.indices .= 0
    end

    output = layer.output
    indices = layer.indices

    for c in 1:input_channels
        for i in 1:output_height
            for j in 1:output_width
                i_start = (i - 1) * pool_height + 1
                i_end = i_start + pool_height - 1
                j_start = (j - 1) * pool_width + 1
                j_end = j_start + pool_width - 1

                patch = input[i_start:i_end, j_start:j_end, c, :]
                max_val = maximum(patch, dims=(1, 2))

                output[i, j, c, :] = max_val
                indices[i_start:i_end, j_start:j_end, c, :] .= (patch .== max_val)
            end
        end
    end

    return output
end

function backward_pass(layer::MaxPoolLayer, dL_dY::Array)
    dL_dX = zeros(size(layer.input))
    pool_height, pool_width = layer.pool_size
    input_height, input_width, input_channels, batch_size = size(layer.input)

    for c in 1:input_channels
        for i in 1:size(dL_dY, 1)
            for j in 1:size(dL_dY, 2)
                i_start = (i - 1) * pool_height + 1
                i_end = i_start + pool_height - 1
                j_start = (j - 1) * pool_width + 1
                j_end = j_start + pool_width - 1

                for b in 1:batch_size
                    dL_dX[i_start:i_end, j_start:j_end, c, b] = dL_dY[i, j, c, b] * layer.indices[i_start:i_end, j_start:j_end, c, b]
                end
            end
        end
    end

    return dL_dX
end

function update_weights(layer::Layer, learning_rate::Float64, batch_size::Int64)

end