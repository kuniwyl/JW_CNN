using MLDatasets, Flux

# Load the MNIST data 
train_data = MLDatasets.MNIST(split=:train);
test_data  = MLDatasets.MNIST(split=:test);

function relu(x)
    return max.(0, x)
end

function relu_derivative(x)
    return x .> 0
end

function identity(x)
    return x
end

function identity_derivative(x)
    return ones(size(x))
end

include("src/JW_CNN.jl")
network = JW_CNN.NeuralNetwork(0.1, 200);
JW_CNN.add_layer!(network, JW_CNN.ConvLayer(3, 3, 6));
JW_CNN.add_layer!(network, JW_CNN.MaxPoolLayer(2, 2));
JW_CNN.add_layer!(network, JW_CNN.ConvLayer(3, 3, 16));
JW_CNN.add_layer!(network, JW_CNN.MaxPoolLayer(2, 2));
JW_CNN.add_layer!(network, JW_CNN.FlattenLayer());
JW_CNN.add_layer!(network, JW_CNN.FCLayer(400, 84, relu, relu_derivative));
JW_CNN.add_layer!(network, JW_CNN.FCLayer(84, 10, identity, identity_derivative));

inputs = reshape(train_data.features, 28, 28, 1, :);
targets = Flux.onehotbatch(train_data.targets, 0:9);
targets = reshape(targets, 10, :);

test_input = reshape(test_data.features, 28, 28, 1, :);
test_targets = Flux.onehotbatch(test_data.targets, 0:9);
test_targets = reshape(test_targets, 10, :);

# small_input = inputs[:, :, :, 1:1000];
# small_targets = targets[:, 1:1000];
# @time JW_CNN.train(network, small_input, small_targets, test_input, test_targets, 3)

# JW_CNN.test(network, test_input, test_targets)
@time JW_CNN.train(network, inputs, targets, test_input, test_targets, 2)
@time JW_CNN.test(network, test_input, test_targets)  