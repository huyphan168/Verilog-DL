import torch
cnn_weights = torch.load("cnn_weights.pth")

# Extract the values of the filters and biases for the convolutional layer
filters = cnn_weights["conv1.weight"]
biases = cnn_weights["conv1.bias"]

# Convert the filters and biases to a binary file
with open("weights.bin", "wb") as f:
    f.write(filters.tobytes())
    f.write(biases.tobytes())
