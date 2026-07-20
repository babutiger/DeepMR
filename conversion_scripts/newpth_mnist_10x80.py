# Convert an ordinary trained pth file into the pth file required by beta-crown 2025-01-14 21:42:54

# Load the pretrained model weights: after instantiating the OriginalNeuralNetwork class, make sure you load the corresponding weights.
# Avoid re-initializing the weights every time the code runs: load the saved model weights directly from a file.
# Automatically generate the mapping dictionary

import torch
from torch import nn

# Definition of the original model
class OriginalNeuralNetwork(nn.Module):
    def __init__(self):
        super(OriginalNeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 10)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

# Modified model, defined as nn.Sequential with nn.Flatten() added
def create_modified_model():
    model = nn.Sequential(
        nn.Flatten(),  # Add a Flatten layer
        nn.Linear(784, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 10)
    )
    return model


# Load the original model parameters
original_model_path = '../mnist_net_new_10x80.pth'
original_state_dict = torch.load(original_model_path)

# Create an instance of the original model and load the weights
old_model = OriginalNeuralNetwork()
old_model.load_state_dict(original_state_dict)

# Create a new model instance
new_model = create_modified_model()

# Automatically generate the mapping dictionary
mapping = {}
for i, layer in enumerate(old_model.linear_relu_stack):
    if isinstance(layer, nn.Linear):
        print(f"i:{i}")
        # This was originally wrong, so some layers were not handled and only 0 4 8 were processed, because I multiplied by 2 while i itself is already 0 2 4 2025-01-14 21:14:23
        mapping[f'linear_relu_stack.{i}.weight'] = f'{i + 1}.weight'
        mapping[f'linear_relu_stack.{i}.bias'] = f'{i + 1}.bias'



# Convert the original model parameters to the new model format
new_state_dict = {}
for old_key in mapping.keys():
    new_key = mapping[old_key]
    if old_key in original_state_dict:
        new_state_dict[new_key] = original_state_dict[old_key]



# new_state_dict = {}
# # Rename parameters according to the mapping relationship
# for old_key, new_key in mapping.items():
#     if old_key in original_state_dict:
#         new_state_dict[new_key] = original_state_dict[old_key]


# Load the corrected parameters into the new model
new_model.load_state_dict(new_state_dict, strict=False)

# Save the new model as a .pth file
new_model_path = '../mnist_net_new_10x80_check_for_bcrown.pth'
torch.save(new_model.state_dict(), new_model_path)

print(f"New model saved to {new_model_path} with nn.Flatten() added.")

# Print the structure and parameter information of the original model
print("\nOriginal Model Structure:")
print(old_model)

print("\nOriginal Model Layer-wise Parameters:")
for name, param in old_model.named_parameters():
    print(f"{name}: {param.size()}")

# Print the modified model structure
print("\nModified Model Structure:")
print(new_model)

# Print the dimensions of each layer's parameters in the modified model
print("\nModified Model Layer-wise Parameters:")
for name, param in new_model.named_parameters():
    print(f"{name}: {param.size()}")

# Print all weight and bias parameters of the old and new models
print("\nOriginal Model Weights and Biases:")
for name, param in old_model.named_parameters():
    if 'weight' in name or 'bias' in name:
        print(f"{name}: {param.data}")

print("\nModified Model Weights and Biases:")
for name, param in new_model.named_parameters():
    if 'weight' in name or 'bias' in name:
        print(f"{name}: {param.data}")
