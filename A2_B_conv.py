import torch
import pytorch_lightning as pl

class ConvolutionalNN(pl.LightningModule):
    def __init__(self, model, loss_function, accuracy_function, optimizer_function, optimizer_params):
        super().__init__()

        # Store accuracy metric for training and validation
        self.train_accuracy = accuracy_function
        self.val_accuracy = accuracy_function

        # Store the loss function and model
        self.loss_function = loss_function
        self.model = model

        # Initialize the optimizer using the provided optimizer function and its parameters
        self.optimizer = optimizer_function(self.model.parameters(), **optimizer_params)
        
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training logic for a single batch
        x, y = batch
        y_hat = self.forward(x)                         # Get model predictions
        loss = self.loss_function(y_hat, y)             # Compute loss

        # Update training accuracy
        self.train_accuracy(torch.argmax(y_hat, dim=1), y)

        # Log training loss and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        # Reset training accuracy metric at the end of the epoch
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        # Validation logic for a single batch
        x, y = batch
        y_hat = self.forward(x)                         # Get model predictions
        loss = self.loss_function(y_hat, y)             # Compute loss

        # Update validation accuracy
        self.val_accuracy(torch.argmax(y_hat, dim=1), y)

        # Log validation metrics (on_epoch=True ensures it's averaged over the epoch)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):
        # Log final validation accuracy at the end of the epoch
        self.log('val_accuracy', self.val_accuracy.compute())
        # Reset the validation accuracy metric
        self.val_accuracy.reset()

    def configure_optimizers(self):
        # Return the optimizer for training
        return self.optimizer
    
    def test_step(self, batch, batch_idx):
        # Testing logic for a single batch
        x, y = batch
        y_hat = self.forward(x)                         # Get model predictions
        loss = self.loss_function(y_hat, y)             # Compute loss

        # Update accuracy metric (same metric used for test as val)
        self.val_accuracy.update(torch.argmax(y_hat, dim=1), y)

        # Log test accuracy and loss ,only on epoch level
        self.log("test_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss
