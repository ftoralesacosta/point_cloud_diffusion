# JUST A SKELETON FOR NOW
import torch
import torch.optim as optim
import sys
import yaml
sys.path.insert(1, '../models/')
from point_cloud_diffusion import PCD
# import diffusion_PyTorch

# Assuming the PCD class is already defined
config_file = "../configs/default_config.yaml"
model = PCD(config_file=config_file)  # Initialize your model
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use the Adam optimizer
criterion = torch.nn.MSELoss()  # Example loss function, change if necessary

# Placeholder for your training data and labels
# You will need to replace these with your actual data loading mechanism
train_data = torch.rand(100, 1)  # Example input data
train_labels = torch.rand(100, 1)  # Example labels/targets

# Number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    optimizer.zero_grad()  # Clear the gradients

    # Forward pass: Compute the model output for the input data

    # exit("Not Ready model training yet!!!")
    outputs = model(train_data)

    # Compute the loss
    loss = criterion(outputs, train_labels)

    # Backward pass: Compute gradient of the loss with respect to model parameters
    loss.backward()

    # Perform a single optimization step (parameter update)
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# After training, you can save the model if needed
torch.save(model.state_dict(), 'gsgm_model.pth')


