import torchmetrics
from A2_B_conv import ConvolutionalNN
import torch
import torchvision as tv
import pytorch_lightning as pl
from operator import attrgetter

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the iNaturalist dataset using ImageFolder and apply transformations
dataset = tv.datasets.ImageFolder(
    root='inaturalist_12K/train',
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),                # Convert images to PyTorch tensors
        tv.transforms.Resize((300, 300)),        # Resize images to 300x300
        # You can normalize or push to device here if needed
    ]),
)

# Split the dataset equally into training and validation sets
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])

# ------------------------------
# Load a pretrained ViT model
# ------------------------------

# Specify the model name and weights from torchvision
model_name = 'ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1'
model_weights = attrgetter(model_name)(tv.models)                          # Fetch the weight enum
model_architecture = attrgetter(model_name.split('_Weights')[0].lower())(tv.models)  # Get model constructor

# Load the pretrained model with the selected weights
model = model_architecture(weights=model_weights)

# Replace the final classification head to match our 10-class dataset
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Freeze all layers in the model (useful for feature extraction)
model_layers = list(model.children())
[layer.requires_grad_(False) for layer in model_layers]

# Unfreeze only the final classification layer
model.fc.requires_grad_(True)

# Print model architecture (optional)
print(model)

# ------------------------------
# Setup training components
# ------------------------------

# Define optimizer and hyperparameters
optimizer_function = torch.optim.Adam
optimizer_params = {}

# Define accuracy metric for multiclass classification (10 classes)
accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)

# Define loss function
loss_function = torch.nn.CrossEntropyLoss()

# Wrap everything into the PyTorch Lightning module
model = ConvolutionalNN(
    model=model,
    loss_function=loss_function,
    accuracy_function=accuracy_function,
    optimizer_function=optimizer_function,
    optimizer_params=optimizer_params
)

# ------------------------------
# Training setup
# ------------------------------

# Define the PyTorch Lightning trainer
trainer = pl.Trainer(
    log_every_n_steps=5,  # Log metrics every 5 steps
    max_epochs=10         # Train for 10 epochs
)

# Prepare DataLoaders
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=int(len(train_data)/3))
val_dataloaders = torch.utils.data.DataLoader(val_data)

# model training
trainer.fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
