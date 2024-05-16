mutable struct ConvLayer <: Layer
    filters::Array{Float32, 3}
    filterHeight::Int
    filterWidth::Int
    filtersNum::Int
    biases::Array{Float32, 1}

    input::Array{Float32, 4}
    output::Array{Float32, 4}
    outputSize::Tuple{Int, Int, Int, Int}

    activation::Function
    deActivation::Function
    
    dL_dW::Array{Float32, 3}
    dL_dB::Array{Float32, 1}
    dL_dX::Array{Float32, 4}
end


function relu(x::Array{Float32, 4})::Array{Float32, 4}
    return max.(0, x)
end


function relu_derivative(x::Array{Float32, 4})::Array{Float32, 4}
    return x .> 0
end


function ConvLayer(filter_height::Int, filter_width::Int, num_filters::Int)::ConvLayer
    filters = randn(filter_height, filter_width, num_filters)
    biases = zeros(num_filters)

    dL_dW = zeros(size(filters))
    dL_dB = zeros(size(biases))
    dL_dX = zeros(1, 1, 1, 1)

    input = zeros(1, 1, 1, 1)
    output = zeros(1, 1, 1, 1)

    activation = relu
    deActivation = relu_derivative
    
    return ConvLayer(filters, filter_height, filter_width, num_filters, biases, input, output, (1, 1, 1, 1), activation, deActivation, dL_dW, dL_dB, dL_dX)
end


function forward_pass(layer::ConvLayer, input::Array)::Array{Float32, 4}
    if size(layer.output) == (1, 1, 1, 1)
        inputHeight, inputWidth, _, batchSize = size(input)
        outputHeight = inputHeight - layer.filterHeight + 1
        outputWidth = inputWidth - layer.filterWidth + 1
        layer.outputSize = (outputHeight, outputWidth, layer.filtersNum, batchSize)
        layer.output = zeros(layer.outputSize)
    end

    fill!(layer.output, 0)
    layer.input = input

    @inbounds @views for c in 1:size(input)[4]
        for f in 1:layer.filtersNum
            filter = layer.filters[:, :, f]

            for y in 1:layer.outputSize[1]
                for x in 1:layer.outputSize[2]
                    inputSlice = input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, c]
                    
                    for yy in 1:layer.filterHeight
                        for xx in 1:layer.filterWidth
                            for i in 1:size(input)[3]
                                layer.output[y, x, f, c] += inputSlice[yy, xx, i] * filter[yy, xx]
                            end
                        end
                    end 
                    layer.output[y, x, f, c] += layer.biases[f]
                end
            end
        end
    end
    
    return layer.activation(layer.output)
end


function backward_pass(layer::ConvLayer, dL_dY::Array)::Array{Float32, 4}
    dL_dZ = layer.deActivation(layer.output) .* dL_dY
    input = layer.input
        
    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(input))
    else 
        fill!(layer.dL_dX, 0)
    end

    #calculate weights gradients
    @inbounds for c in 1:size(input)[4]
        for f in 1:layer.filtersNum
            filter = @view layer.filters[:, :, f]
            for y in 1:layer.outputSize[1]
                for x in 1:layer.outputSize[2]
                    dL_dZ_slice = dL_dZ[y, x, f, c]
                    if (dL_dZ_slice != 0)
                        for i in 1:size(input)[3]
                            inputSlice = @view input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, i, c]
                            for yy in 1:layer.filterHeight
                                for xx in 1:layer.filterWidth
                                    layer.dL_dW[yy, xx, f] += inputSlice[yy, xx] * dL_dZ_slice
                                    layer.dL_dX[y+yy-1, x+xx-1, i, c] += filter[yy, xx] * dL_dZ_slice
                                end
                            end
                        end
                    end
                end
            end
            layer.dL_dB[f] += sum(dL_dZ[:, :, f, c])
        end
    end

    return layer.dL_dX
end


function update_weights(layer::ConvLayer, learning_rate::Float64, batch_size::Int64)
    layer.filters .-= learning_rate * (layer.dL_dW / batch_size)
    layer.biases .-= learning_rate * (layer.dL_dB / batch_size)

    # Reset the gradients
    fill!(layer.dL_dW, 0)
    fill!(layer.dL_dB, 0)

    return
end