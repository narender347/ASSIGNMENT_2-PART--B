import argparse
from operator import attrgetter
import numpy as np
import torchmetrics
from A2_Conv import ConvolutionalNN
import torch
import torchvision as tv
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import os

def main(raw_args=[]):
    print('------- Now begins training -------')

    # -------------------------
    # Argument Parsing
    # -------------------------
    parser = argparse.ArgumentParser(description='CNN Training Configuration')

    # Weights & Biases (wandb) configuration
    parser.add_argument('-wandb_project', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-wandb_entity', '--wandb_entity', type=str, default='myname',
                        help='Wandb entity (username/team) for logging experiments')
    parser.add_argument('-wandb_sweepid', '--wandb_sweepid', type=str, default=None,
                        help='Wandb Sweep ID (used during hyperparameter sweeps)')

    # Dataset and training configuration
    parser.add_argument('-dataset', '--dataset', type=str, default='inaturalist_12K',
                        choices=["inaturalist_12K"], help='Dataset choices')

    parser.add_argument('-epochs', '--epochs', type=int, default=10,
                        help='Number of epochs to train the model')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=32,
                        help='Batch size for training')

    # Model architecture configuration
    parser.add_argument('-model_name_weights', '--model_name_weights', type=str,
                        default='ResNet18_Weights.IMAGENET1K_V1',
                        help='Pretrained model weights name from torchvision.models')

    # Layer unfreezing options
    parser.add_argument('-unfreeze_first_layers', '--unfreeze_first_layers', type=int, default=1,
                        help='Number of layers to unfreeze from the start')
    parser.add_argument('-unfreeze_last_layers', '--unfreeze_last_layers', type=int, default=1,
                        help='Number of layers to unfreeze from the end')

    # Dense layer customization (not used in current implementation)
    parser.add_argument('-num_dense_layers', '--num_dense_layers', type=int, default=1,
                        help='Number of dense layers in the model')
    parser.add_argument('-dense_change_ratio', '--dense_change_ratio', type=int, default=4,
                        help='Dense layer size change ratio')

    # Parse CLI arguments
    args = parser.parse_args(raw_args)

    # -------------------------
    # Device Setup
    # -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # -------------------------
    # Dataset Preparation
    # -------------------------
    image_transform = [
        tv.transforms.ToTensor(),                            # Convert image to tensor
        tv.transforms.Resize((500, 300)),                    # Resize image to 500x300
        tv.transforms.Normalize((0.5, 0.5, 0.5),              # Normalize with mean and std dev
                               (0.5, 0.5, 0.5)),
    ]

    # Load dataset and apply transformations
    dataset = tv.datasets.ImageFolder(root='inaturalist_12K/train',
                                      transform=tv.transforms.Compose(image_transform))

    # Split dataset into training and validation sets (80-20)
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # -------------------------
    # Load Pretrained Model
    # -------------------------
    model_name = args.model_name_weights
    model_weights = attrgetter(model_name)(tv.models)  # e.g., ResNet18_Weights.IMAGENET1K_V1
    model_architecture = attrgetter(model_name.split('_Weights')[0].lower())(tv.models)

    # Instantiate the model with weights
    model = model_architecture(weights=model_weights)

    # Replace final fully connected layer to output 10 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # -------------------------
    # Freeze / Unfreeze Layers
    # -------------------------
    model_layers = list(model.children())

    # Freeze all layers initially
    [layer.requires_grad_(False) for layer in model_layers]

    # Unfreeze last N layers (from the end)
    for layer_index in range(-args.unfreeze_last_layers, 0):
        model_layers[layer_index].requires_grad_(True)

    # Unfreeze first N layers (from the start)
    for layer_index in range(args.unfreeze_first_layers):
        model_layers[layer_index].requires_grad_(True)

    # -------------------------
    # Training Setup
    # -------------------------
    optimizer_function = torch.optim.NAdam               # Using NAdam optimizer
    optimizer_params = {}                                # Default params
    accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    loss_function = torch.nn.CrossEntropyLoss()          # Suitable for multi-class classification

    # Wrap the model into a PyTorch Lightning module
    model = ConvolutionalNN(
        model=model,
        loss_function=loss_function,
        accuracy_function=accuracy_function,
        optimizer_function=optimizer_function,
        optimizer_params=optimizer_params
    )

    # -------------------------
    # Weights & Biases Logging
    # -------------------------
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, reinit=True)

    # Log model architecture and gradients
    wandb_logger.watch(model)

    # -------------------------
    # Train the Model
    # -------------------------
    trainer = pl.Trainer(
        log_every_n_steps=5,              # Frequency of logging to wandb
        max_epochs=args.epochs,           # Total number of epochs
        logger=wandb_logger               # wandb integration
    )

    # DataLoaders for training and validation
    train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    val_dataloaders = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)

    # Start training
    trainer.fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)

if __name__ == '__main__':
    main()
