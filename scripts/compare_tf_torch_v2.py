import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TensorFlow function
class TFEmbedding:
    def __init__(self, num_embed, projection_dim):
        self.num_embed = num_embed
        self.dense1 = layers.Dense(2 * num_embed, activation=None, use_bias=True)
        self.dense2 = layers.Dense(num_embed, use_bias=True)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.01)

    def Embedding(self, inputs, projection):
        angle = inputs * projection
        embedding = tf.concat([tf.math.sin(angle), tf.math.cos(angle)], -1)
        embedding = self.dense1(embedding)
        embedding = self.activation(embedding)
        embedding = self.dense2(embedding)
        return embedding

# PyTorch class
class TorchEmbedding(nn.Module):
    def __init__(self, projection_dim, num_embed):
        super(TorchEmbedding, self).__init__()
        self.dense1 = nn.Linear(2 * projection_dim, 2 * num_embed)
        self.dense2 = nn.Linear(2 * num_embed, num_embed)

    def forward(self, inputs, projection):
        angle = inputs * projection
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        embedding = self.dense1(embedding)
        embedding = F.leaky_relu(embedding, 0.01)
        embedding = self.dense2(embedding)
        return embedding

def copy_weights_tf_to_torch(tf_model, torch_model):
    # Copy weights from TensorFlow to PyTorch
    torch_model.dense1.weight.data = torch.tensor(tf_model.dense1.weights[0].numpy().T)
    torch_model.dense1.bias.data = torch.tensor(tf_model.dense1.weights[1].numpy())
    
    torch_model.dense2.weight.data = torch.tensor(tf_model.dense2.weights[0].numpy().T)
    torch_model.dense2.bias.data = torch.tensor(tf_model.dense2.weights[1].numpy())

# Parameters for the test
num_embed = 10
projection_dim = 5
input_dim = 5  # Assume same as projection_dim for simplicity

# Initialize the models
tf_model = TFEmbedding(num_embed, projection_dim)
torch_model = TorchEmbedding(projection_dim, num_embed)

# Synchronize weights and biases
tf_model.Embedding(np.zeros((1, input_dim)), np.zeros((1, input_dim)))  # Initialize TensorFlow weights
copy_weights_tf_to_torch(tf_model, torch_model)

# Generate random input and projection
np.random.seed(0)  # for reproducibility
inputs = np.random.rand(1, input_dim).astype(np.float32)
projection = np.random.rand(1, input_dim).astype(np.float32)

# Get the TensorFlow output
tf_output = tf_model.Embedding(inputs, projection)

# Convert inputs to torch tensors and get the PyTorch output
torch_inputs = torch.tensor(inputs)
torch_projection = torch.tensor(projection)
torch_output = torch_model(torch_inputs, torch_projection)

# Compare the outputs
# Convert the PyTorch output to numpy for comparison
torch_output_np = torch_output.detach().numpy()

# Calculate the difference between TensorFlow and PyTorch outputs
difference = np.abs(tf_output - torch_output_np)
mean_difference = np.mean(difference)

print("Mean difference between TensorFlow and PyTorch outputs:", mean_difference)

