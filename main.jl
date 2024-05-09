using MLDatasets, Flux

# Load the MNIST data 
train_data = MLDatasets.MNIST(split=:train);
test_data  = MLDatasets.MNIST(split=:test);

include("src/JW_CNN.jl")
network = JW_CNN.NeuralNetwork(0.1);
# JW_CNN.add_layer!(network, JW_CNN.ConvLayer(3, 3, 6));
# JW_CNN.add_layer!(network, JW_CNN.MaxPoolLayer(2, 2));
# JW_CNN.add_layer!(network, JW_CNN.ConvLayer(3, 3, 16));
# JW_CNN.add_layer!(network, JW_CNN.MaxPoolLayer(2, 2));
JW_CNN.add_layer!(network, JW_CNN.FlattenLayer());
JW_CNN.add_layer!(network, JW_CNN.FCLayer(784, 84));
JW_CNN.add_layer!(network, JW_CNN.FCLayer(84, 10));

inputs = reshape(train_data.features, 784, 1, 1, :);
targets = Flux.onehotbatch(train_data.targets, 0:9);
targets = reshape(targets, 10, :);

test_input = reshape(test_data.features, 784, 1, 1, :);
test_targets = Flux.onehotbatch(test_data.targets, 0:9);
test_targets = reshape(test_targets, 10, :);

# JW_CNN.test(network, test_input, test_targets)
JW_CNN.train(network, inputs, targets, 15, 200)
JW_CNN.test(network, test_input, test_targets) 