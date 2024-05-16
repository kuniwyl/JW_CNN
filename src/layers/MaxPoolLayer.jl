mutable struct MaxPoolLayer <: Layer
    pool_size::Tuple{Int, Int}
    
    input::Array{Float32, 4}
    output::Array{Float32, 4}
    outputSize::Tuple{Int, Int, Int, Int}
    indices::Array{Float32, 4}

    i_starts::StepRange{Int, Int}
    i_ends::StepRange{Int, Int}
    j_starts::StepRange{Int, Int}
    j_ends::StepRange{Int, Int}

    dL_dX::Array{Float32, 4}
end

function MaxPoolLayer(height::Int, width::Int)::MaxPoolLayer
    input = zeros(1, 1, 1, 1)
    output = zeros(1, 1, 1, 1)
    outputSize = (1, 1, 1, 1)
    indices = zeros(1, 1, 1, 1)

    dL_dX = zeros(1, 1, 1, 1)
    
    i_start = 1:1:1
    i_end = 1:1:1
    j_start = 1:1:1
    j_end = 1:1:1

    return MaxPoolLayer((height, width), input, output, outputSize, indices, i_start, i_end, j_start, j_end, dL_dX)
end

function forward_pass(layer::MaxPoolLayer, input::Array)
    if size(layer.output) == (1, 1, 1, 1)
        pool_height, pool_width = layer.pool_size
        input_height, input_width, input_channels, batch_size = size(input)
        layer.input = zeros(size(input))
        output_height = div(input_height, pool_height)
        output_width = div(input_width, pool_width)
        layer.output = zeros(output_height, output_width, input_channels, batch_size)
        layer.indices = zeros(input_height, input_width, input_channels, batch_size)
        layer.i_starts = 1:pool_height:input_height
        layer.i_ends = pool_height:pool_height:input_height
        layer.j_starts = 1:pool_width:input_width
        layer.j_ends = pool_width:pool_width:input_width
    end

    layer.input = input
    fill!(layer.output, 0)

    @inbounds @views for i in 1:size(layer.output)[1]
        for j in 1:size(layer.output)[2]
            i_start = layer.i_starts[i]
            i_end = layer.i_ends[i]
            j_start = layer.j_starts[j]
            j_end = layer.j_ends[j]

            for c in 1:size(input)[3]
                for b in 1:size(input)[4]
                    @views window = input[i_start:i_end, j_start:j_end, c, b]
                    layer.output[i, j, c, b] = maximum(window)
                    layer.indices[i_start:i_end, j_start:j_end, c, b] .= window .== layer.output[i, j, c, b]
                end
            end
        end
    end

    return layer.output
end

function backward_pass(layer::MaxPoolLayer, dL_dY::Array)
    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(layer.input))
    end
    fill!(layer.dL_dX, 0)

    @inbounds @views for i in 1:size(layer.output)[1]
        for j in 1:size(layer.output)[2]
            i_start = layer.i_starts[i]
            i_end = layer.i_ends[i]
            j_start = layer.j_starts[j]
            j_end = layer.j_ends[j]

            for c in 1:size(layer.output)[3]
                for b in 1:size(layer.output)[4]
                    dL_dY_slice = dL_dY[i, j, c, b]
                    
                    if dL_dY_slice != 0
                        indices = layer.indices[i_start:i_end, j_start:j_end, c, b]
                        for x in 1:size(indices)[1]
                            for y in 1:size(indices)[2]
                                if indices[x, y] != 0
                                    layer.dL_dX[x, y, c, b] = dL_dY_slice
                                end
                            end
                        end 
                    end
                end
            end
        end
    end

    return layer.dL_dX
end

function update_weights(layer::Layer, learning_rate::Float64, batch_size::Int64)

end