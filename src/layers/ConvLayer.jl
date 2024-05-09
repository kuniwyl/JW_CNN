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
    layer.input = input

    if size(layer.output) == (1, 1, 1, 1)
        inputHeight, inputWidth, _, batchSize = size(input)
        outputHeight = inputHeight - layer.filterHeight + 1
        outputWidth = inputWidth - layer.filterWidth + 1
        layer.outputSize = (outputHeight, outputWidth, layer.filtersNum, batchSize)
        layer.output = zeros(layer.outputSize)
    end
    fill!(layer.output, 0)
    
    for f in 1:layer.outputSize[3]
        filter = layer.filters[:, :, f]

        for y in 1:layer.outputSize[1]
            for x in 1:layer.outputSize[2]
                result = input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, :] .* filter
                layer.output[y, x, f, :] += vec(sum(result, dims=(1, 2, 3))) .+ layer.biases[f]
            end
        end
    end

    return layer.activation(layer.output)
end

function forward_pass_single(layer::ConvLayer, input::Array{Float32, 3}, batchId)
    for f in 1:layer.filtersNum
        filter = layer.filters[:, :, f]

        for y in 1:layer.outputSize[1]
            for x in 1:layer.outputSize[2]
                layer.output[y, x, f, batchId] += sum(input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :] .* filter) + layer.biases[f]
            end
        end
    end
end

function forward_pass!(layer::ConvLayer, input::Array) 
    if size(layer.output) == (1, 1, 1, 1)
        inputHeight, inputWidth, _, batchSize = size(input)
        outputHeight = inputHeight - layer.filterHeight + 1
        outputWidth = inputWidth - layer.filterWidth + 1
        layer.outputSize = (outputHeight, outputWidth, layer.filtersNum, batchSize)
        layer.output = zeros(layer.outputSize)
    end
    fill!(layer.output, 0)

    @threads for i in 1:layer.outputSize[4]
        forward_pass_single(layer, input[:, :, :, i], i)
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

    # Iterate over each filter
    for f in 1:layer.filtersNum
        filter = layer.filters[:, :, f]
        for y in 1:layer.outputSize[1]
            for x in 1:layer.outputSize[2]
                dL_dZ_slice = dL_dZ[y, x, f, :]
                layer.dL_dB[f] += sum(dL_dZ_slice)

                for i in 1:size(input)[4]
                    layer.dL_dW[:, :, f] .+= sum(input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, i] .* dL_dZ_slice[i], dims=3)
                    layer.dL_dX[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, i] .+= sum(filter .* dL_dZ_slice[i], dims=3)
                end
            end
        end
    end

    return layer.dL_dX
end

function backward_pass!(layer::ConvLayer, dL_dY::Array, input::Array)::Array{Float32, 4}
    dL_dZ = layer.deActivation(layer.output) .* dL_dY

    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(input))
    else 
        fill!(layer.dL_dX, 0)
    end

    for f in 1:layer.filtersNum
        filter = layer.filters[:, :, f]

        for y in 1:layer.outputSize[1]
            for x in 1:layer.outputSize[2]
                dL_dZ_slice = dL_dZ[y, x, f, :]
                layer.dL_dB[f] += sum(dL_dZ_slice)

                for i in 1:size(input)[4]
                    layer.dL_dW[:, :, f] .+= sum(input[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, i] .* dL_dZ_slice[i], dims=3)
                    layer.dL_dX[y:y+layer.filterHeight-1, x:x+layer.filterWidth-1, :, i] .+= sum(filter .* dL_dZ_slice[i], dims=3)
                end
            end
        end
    end

    return layer.dL_dX
end

function backward_pass_single(layer::ConvLayer, dL_dY::Array{Float32, 3}, input::Array{Float32, 3}, batchId::Int64)
    for f in 1:layer.filtersNum
        filter = layer.filters[:, :, f]

        for y in 1:layer.outputSize[1]
            for x in 1:layer.outputSize[2]
                dL_dZ_slice = dL_dY[y, x, f]
                layer.dL_dB[f] += sum(dL_dZ_slice)

                y_end = y + layer.filterHeight - 1
                x_end = x + layer.filterWidth - 1
                layer.dL_dW[:, :, f] .+= sum(input[y:y_end, x:x_end, :] .* dL_dZ_slice, dims=3)
                layer.dL_dX[y:y_end, x:x_end, :, batchId] .+= sum(filter .* dL_dZ_slice, dims=3)
            end
        end
    end
end


function update_weights(layer::ConvLayer, learning_rate::Float64, batch_size::Int64)
    layer.filters .-= learning_rate * (layer.dL_dW / batch_size)
    layer.biases .-= learning_rate * (layer.dL_dB / batch_size)

    # Reset the gradients
    fill!(layer.dL_dW, 0)
    fill!(layer.dL_dB, 0)

    return
end