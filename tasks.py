import numpy as np

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
import torch.optim as optim

import wandb



def train_model(
        model, 
        pl_module,
        epochs, 
        train_loader, 
        test_loader, 
        model_hparams, 
        device="cuda", 
        log_wandb=False, 
        **kwargs):
    """
    
    """

    if log_wandb:
        logger = WandbLogger(
            project="WSConv",
            group=pl_module.__name__,
            name=model.__class__.__name__,
            entity="computri",
            config=model_hparams
        )
    else:
        logger = None

    print(model)

    callbacks = [pl.callbacks.LearningRateMonitor("epoch")]

    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(accelerator="gpu" if device == "cuda" else "cpu",      # We run on a single GPU (if possible)
                         max_epochs=epochs,       # How many epochs to train for if no patience is set
                         logger=logger,
                         callbacks=callbacks)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = pl_module(model=model, **kwargs)
    
    trainer.fit(model, train_loader, test_loader)

    if log_wandb:
        wandb.finish()

    return model, logger


class Classification(pl.LightningModule):

    def __init__(
        self, 
        model, 
        optimizer_name, 
        optimizer_hparams, 
        include_model_loss=False, 
        norm_scaler=1.0,
        entropy_scaler=0.01, 
        visualize=False,
        num_classes=10,
        lr_reps=None):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model

        self.model_loss = include_model_loss
        self.norm_scaler = norm_scaler
        self.entropy_scaler = entropy_scaler
        if lr_reps is None:
            self.lr_reps = self.hparams.optimizer_hparams["lr"]
        else:
            self.lr_reps = lr_reps
        self.num_classes = num_classes
        self.visualize = visualize
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        
        param_list = []

        

        for name, param in self.model.named_parameters():
            if "representation" in name:
                param_list.append({"params": param, "lr": self.lr_reps})
            else:
                param_list.append({"params": param, "lr": self.hparams.optimizer_hparams["lr"]})
        optimizer = optim.AdamW(
            param_list, **self.hparams.optimizer_hparams)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()


        if self.model_loss:
            norm_loss, entropy_loss = self.model.get_loss()

            loss += self.entropy_scaler * entropy_loss + self.norm_scaler * norm_loss
            self.log('entropy_loss', entropy_loss)
            self.log('norm_loss', norm_loss)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('entropy_scaler', self.entropy_scaler, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        if batch_idx == 0 and self.visualize:

            fig, ax = plt.subplots(1, 3)

            ax[0].imshow(imgs[0].squeeze().cpu().numpy())
            ax[1].imshow(imgs[1].squeeze().cpu().numpy())
            ax[2].imshow(imgs[2].squeeze().cpu().numpy())


            self.logger.experiment.log({"train_samples": wandb.Image(fig)})
            plt.close(fig)
            plt.clf()


        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # By default logs it per epoch (weighted average over batches)
        self.log('val/acc', acc, prog_bar=True)

        if batch_idx == 0 and self.visualize:
            self.model.visualize(logger=self.logger)


    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test/acc', acc, prog_bar=True)

    
