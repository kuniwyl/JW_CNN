mutable struct ConvLayer <: Layer
    filters::Array{Float32, 3}
    biases::Array{Float32, 1}

    input::Array{Float32, 4}
    output::Array{Float32, 4}

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
    
    return ConvLayer(filters, biases, input, output, activation, deActivation, dL_dW, dL_dB, dL_dX)
end


function forward_pass(layer::ConvLayer, input::Array)::Array{Float32, 4}
    layer.input = input
    inputHeight, inputWidth, _, batchSize = size(input)
    filterHeight, filterWidth, num_filters = size(layer.filters)
    
    # Calculate output dimensions
    outputHeight = inputHeight - filterHeight + 1
    outputWidth = inputWidth - filterWidth + 1

    if size(layer.output) == (1, 1, 1, 1)
        layer.output = zeros(outputHeight, outputWidth, num_filters, batchSize)
    else 
        layer.output .= 0
    end

    output = layer.output

    for f in 1:num_filters
        filter = layer.filters[:, :, f]
        for y in 1:outputHeight
            for x in 1:outputWidth
                result = input[y:y+filterHeight-1, x:x+filterWidth-1, :, :] .* filter
                output[y, x, f, :] += vec(sum(result, dims=(1, 2, 3))) .+ layer.biases[f]
            end
        end
    end

    # Activation Function (ReLU)
    output = layer.activation(output)
    layer.output = output

    return output
end


function backward_pass(layer::ConvLayer, dL_dY::Array)::Array{Float32, 4}
    dL_dZ = layer.deActivation(layer.output) .* dL_dY

    input = layer.input
    filter_height, filter_width, num_filters = size(layer.filters)
    output_height, output_width, _ = size(dL_dY)

    if size(layer.dL_dX) == (1, 1, 1, 1)
        layer.dL_dX = zeros(size(input))
    end
    layer.dL_dX .= 0

    # Iterate over each filter
    for f in 1:num_filters
        filter = layer.filters[:, :, f]
        for y in 1:output_height
            for x in 1:output_width
                dL_dZ_slice = dL_dZ[y, x, f, :]
                layer.dL_dB[f] += sum(dL_dZ_slice)

                for i in 1:size(input)[4]
                    layer.dL_dW[:, :, f] .+= sum(input[y:y+filter_height-1, x:x+filter_width-1, :, i] .* dL_dZ_slice[i], dims=3)
                    layer.dL_dX[y:y+filter_height-1, x:x+filter_width-1, :, i] .+= sum(filter .* dL_dZ_slice[i], dims=3)
                end
            end
        end
    end

    return layer.dL_dX
end


function update_weights(layer::ConvLayer, learning_rate::Float64, batch_size::Int64)
    layer.filters .-= learning_rate * layer.dL_dW / batch_size
    layer.biases .-= learning_rate * layer.dL_dB / batch_size

    # Reset the gradients
    layer.dL_dW .= 0
    layer.dL_dB .= 0
end