using Random
using CSV
using DataFrames

Random.seed!(0)

# Initialize input data, filters, and biases
input_data = rand(4, 4, 2)  # Random data simulating the output of the first conv layer
filters = rand(3, 3, 2, 3)  # 16 filters of size 3x3x6
biases = rand(3)            # One bias per filter

# Print input data, filters, and biases
println("Input data:", input_data)
println("Filters:", filters)
println("Biases:", biases)

# Output dimensions
output_height = 2
output_width = 2
output_feature_maps = zeros(output_height, output_width, 3)

# Forward pass: Apply convolution and add bias
for f in 1:3  # Loop over each filter
    for i in 1:output_height
        for j in 1:output_width
            region = input_data[i:i+2, j:j+2, :]
            output_feature_maps[i, j, f] = sum(region .* filters[:, :, :, f]) + biases[f]
        end
    end
end
println("Output feature maps:", output_feature_maps)

# Hypothetical gradients from the next layer (backpropagation)
grad_output = rand(output_height, output_width, 3)  # Random gradient from next layer
println("Gradient from the next layer:", grad_output)

# Gradients for weights and biases initialization
grad_filters = zeros(size(filters))
grad_biases = zeros(size(biases))

# Backpropagation to compute gradients
for f in 1:3
    for i in 1:output_height
        for j in 1:output_width
            region = input_data[i:i+2, j:j+2, :]
            grad_filters[:, :, :, f] .+= region .* grad_output[i, j, f]
            grad_biases[f] += grad_output[i, j, f]
        end
    end
end
println("Gradient for filters:", grad_filters)
println("Gradient for biases:", grad_biases)

# Learning rate
learning_rate = 0.01

# Update weights and biases
filters .-= learning_rate .* grad_filters
biases .-= learning_rate .* grad_biases

# Print updated filters and biases
println("Updated filters:", filters)
println("Updated biases:", biases)