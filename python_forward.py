import numpy as np

np.random.seed(1)
batch_size = 20000

# generate random int matrix 7x7
matrix = np.random.rand(batch_size, 1, 8, 8)
print(matrix)

#create filter 2x3x3
filters = np.random.rand(2, 3, 3)
print(filters)

# apply filter to matrix
result = np.zeros((batch_size, 2, 6, 6))
for x in range(2):
    filter = filters[x, :, :]
    for i in range(6):
        for j in range(6):
            r = matrix[:, :, i:i+3, j:j+3] * filter
            result[:, x, i, j] += np.sum(r, axis=(1, 2, 3))
print(result)

filters = np.random.rand(3, 3, 3)
print(filters)

# apply filter to matrix
result1 = np.zeros((batch_size, 3, 4, 4))
for x in range(3):
    filter = filters[x, :, :]
    for i in range(4):
        for j in range(4):
            print(np.shape(result[:, :, i:i+3, j:j+3]))
            print(np.shape(filter))
            r = result[:, :, i:i+3, j:j+3] * filter
            result1[:, x, i, j] += np.sum(r, axis=(1, 2, 3))
print(result1)

# max pooling
result2 = np.zeros((batch_size, 3, 2, 2))
for x in range(3):
    for i in range(2):
        for j in range(2):
            r = result1[:, x, i*2:i*2+2, j*2:j*2+2]
            result2[:, x, i, j] = np.max(r, axis=(1, 2))
print(result2)

# flatten
result3 = np.zeros((batch_size, 12))
for x in range(batch_size):
    result3[x] = result2[x].flatten()
print(result3)

# fully connected layer
weights = np.random.rand(12, 10)
print(weights)

result4 = np.dot(result3, weights)
print(result4)

# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

result5 = np.zeros((batch_size, 10))
for x in range(batch_size):
    result5[x, :] = softmax(result4[x, :])
print((result5 * 100).astype(int))

print(len(result5))