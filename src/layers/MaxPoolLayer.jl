mutable struct MaxPoolLayer <: Layer
    pool_size::Tuple{Int, Int}
    
    input::Array{Float32, 4}
    output::Array{Float32, 4}
    outputSize::Tuple{Int, Int, Int, Int}
    indices::Array{Float32, 4}

    dL_dX::Array{Float32, 4}
end

function MaxPoolLayer(height::Int, width::Int)::MaxPoolLayer
    input = zeros(1, 1, 1, 1)
    output = zeros(1, 1, 1, 1)
    outputSize = (1, 1, 1, 1)
    indices = zeros(1, 1, 1, 1)

    dL_dX = zeros(1, 1, 1, 1)
    
    return MaxPoolLayer((height, width), input, output, outputSize, indices, dL_dX)
end

function forward_pass(layer::MaxPoolLayer, input::Array)
    if size(layer.input) == (1, 1, 1, 1)
        layer.input = zeros(size(input))
    end

    layer.input = input
    pool_height, pool_width = layer.pool_size
    input_height, input_width, input_channels, batch_size = size(input)

    output_height = div(input_height, pool_height)
    output_width = div(input_width, pool_width)

    if size(layer.output) == (1, 1, 1, 1)
        layer.output = zeros(output_height, output_width, input_channels, batch_size)
    else
        fill!(layer.output, 0)
    end

    if size(layer.indices) == (1, 1, 1, 1)
        layer.indices = zeros(input_height, input_width, input_channels, batch_size)
    else
        fill!(layer.indices, 0)
    end

    for c in 1:input_channels
        for i in 1:output_height
            for j in 1:output_width
                i_start = (i - 1) * pool_height + 1
                i_end = i_start + pool_height - 1
                j_start = (j - 1) * pool_width + 1
                j_end = j_start + pool_width - 1

                patch = input[i_start:i_end, j_start:j_end, c, :]
                max_val = maximum(patch, dims=(1, 2))

                layer.output[i, j, c, :] = max_val
                layer.indices[i_start:i_end, j_start:j_end, c, :] .= (patch .== max_val)
            end
        end
    end

    return layer.output
end

function forward_pass_single(layer::MaxPoolLayer, input::Array{Float32, 3}, batch::Int64)
    for i in 1:layer.outputSize[1]
        for j in 1:layer.outputSize[2]
            i_start = (i - 1) * layer.pool_size[1] + 1
            i_end = i_start + layer.pool_size[1] - 1
            j_start = (j - 1) * layer.pool_size[2] + 1
            j_end = j_start + layer.pool_size[2] - 1

            for c in 1:layer.outputSize[3]
                patch = input[i_start:i_end, j_start:j_end, c]
                max_val = maximum(patch)

                layer.output[i, j, c, batch] = max_val
                layer.indices[i_start:i_end, j_start:j_end, c, batch] .= (patch .== max_val)
            end
        end
    end
end

function forward_pass!(layer::MaxPoolLayer, input::Array)
    layer.input = input

    if size(layer.output) == (1, 1, 1, 1)
        pool_height, pool_width = layer.pool_size
        output_height = div(size(input, 1), pool_height)
        output_width = div(size(input, 2), pool_width)

        layer.outputSize = (output_height, output_width, size(input, 3), size(input, 4))
        layer.indices = zeros(size(input))
        layer.output = zeros(layer.outputSize)
    else
        fill!(layer.output, 0)
        fill!(layer.indices, 0)
    end

    @threads for i in 1:layer.outputSize[4]
        forward_pass_single(layer, input[:, :, :, i], i)
    end

    return layer.output
end

function backward_pass(layer::MaxPoolLayer, dL_dY::Array)
    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(layer.input))
    end
    fill!(layer.dL_dX, 0)

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
                    layer.dL_dX[i_start:i_end, j_start:j_end, c, b] = dL_dY[i, j, c, b] * layer.indices[i_start:i_end, j_start:j_end, c, b]
                end
            end
        end
    end

    return layer.dL_dX
end

function backward_pass!(layer::MaxPoolLayer, dL_dY::Array, input::Array)
    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(layer.input))
    end
    fill!(layer.dL_dX, 0)

    @threads for i in 1:layer.outputSize[4]
        backward_pass_single(layer, dL_dY[:, :, :, i], i)
    end

    return layer.dL_dX
end

function backward_pass_single(layer::MaxPoolLayer, dL_dY::Array{Float32, 3}, batch::Int)
    for i in 1:size(dL_dY, 1)
        for j in 1:size(dL_dY, 2)
            i_start = (i - 1) * layer.pool_size[1] + 1
            i_end = i_start + layer.pool_size[1] - 1
            j_start = (j - 1) * layer.pool_size[2] + 1
            j_end = j_start + layer.pool_size[2] - 1

            for c in 1:size(dL_dY, 3)
                layer.dL_dX[i_start:i_end, j_start:j_end, c, batch] .+= dL_dY[i, j, c] * layer.indices[i_start:i_end, j_start:j_end, c, batch]
            end
        end
    end
end

function update_weights(layer::Layer, learning_rate::Float64, batch_size::Int64)

end