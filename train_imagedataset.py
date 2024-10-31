import argparse
import torch

import pytorch_lightning as pl

from tasks import train_model, Classification, train_model

from configs.config import get_cfg_defaults
from utils import save_transformed_dataset, get_datasets, update_config_from_args, cfg_to_dict, get_model

from yacs.config import CfgNode as CN


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument('--config', type=str, default="", help='Path to the config file.')

    parser.add_argument('--batch_size', type=int, required=False, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, required=False, help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, required=False, help='random seed')
    parser.add_argument('--learning_rate', type=float, required=False, help='learning rate (default: 0.01)')
    parser.add_argument('--learning_rate_reps', type=float, required=False, help='learning rate for representations (default: 0.01)')
    parser.add_argument('--norm_scaler', type=float, required=False, help='Scaling term on normalization loss')
    parser.add_argument('--entropy_scaler', type=float, required=False, help='Scaling term on entropy loss')
    parser.add_argument('--cuda', type=int, required=False, help=' CUDA training')
    parser.add_argument('--group', type=str, required=False, help='options: so(2), Cn for any n, None')
    parser.add_argument('--transform_labels', type=str, required=False, help='Transform subset or all labels at train time. Options: all, random_n for any n: retrieves random subset of n labels, or a sequence of digits, i.e. 1234')
    parser.add_argument('--log_wandb', type=int, required=False, help='disables wandb_logging')
    parser.add_argument('--n_group', type=int, required=False, help='Size of representation stack')
    parser.add_argument('--hidden_channels', type=int, required=False, help='hidden_channels')
    parser.add_argument('--model', type=str, required=False, help='Which model to use. Options: CNN, GCNN, WSCNN')
    parser.add_argument('--init_mode', type=str, required=False, help='Init for rep stack. Choose rand, rand_log, ones, ones_log, identity')
    parser.add_argument('--visualize', type=int, required=False, help='Visualize')
    parser.add_argument('--augment', type=int, required=False, help='Whether to apply train time augmentation')
    parser.add_argument('--num_layers', type=int, required=False, help='Number of layers')
    parser.add_argument('--kernel_size', type=int, required=False, help='kernel size')
    parser.add_argument('--fix_identity', type=int, required=False, help='fix identity elem')
    parser.add_argument('--n_iter', type=int, required=False, help='Num iterations for sinkhorn operator')
    parser.add_argument('--dataset', type=str, required=False, help='Which dataset to train. Choose: ScaledMNIST, RotatedMNIST, CIFAR10, Galaxy10, CelebA')


    args = parser.parse_args()

    if args.transform_labels is not None:
        args.transform_labels = f'"{args.transform_labels}"'


    # Load default config
    cfg = get_cfg_defaults()


    # Load config file
    if args.config is not None:
        cfg.merge_from_file(args.config)

    for key, val in vars(args).items():
        if key != "config" and val is not None:
          cfg.merge_from_list([key, val])  

    # Convert to a dictionary to allow argparse to update unknown keys
    cfg_dict = cfg_to_dict(cfg)

    if args.transform_labels is not None:
        cfg_dict["transform_labels"] = args.transform_labels
    # Convert dictionary back to CN for consistency
    cfg.merge_from_other_cfg(CN(cfg_dict))
    
    
    pl.seed_everything(cfg.seed)

    device = "cuda" if cfg.cuda else "cpu"

    train_dataset, test_dataset, in_channels, out_channels = get_datasets(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=4
    )

    model, hparams = get_model(
        in_channels=in_channels, 
        out_channels=out_channels, 
        cfg=cfg, 
        device=device
    )
    
    if not "ws" in cfg.model.lower():
    

        model = train_model(
            model=model,
            pl_module=Classification,
            model_hparams=hparams,
            num_classes=out_channels,
            epochs=cfg.epochs,
            optimizer_name="Adam",
            optimizer_hparams={
                "lr": cfg.learning_rate,
                "weight_decay": 1e-4
            },
            log_wandb=cfg.log_wandb,
            train_loader=train_loader,
            test_loader=test_loader
        )

    else:
        model.get_parameter_counts()
        wscnn_model = train_model(
            model=model,
            model_hparams=hparams,
            pl_module=Classification,
            num_classes=out_channels,
            optimizer_name="Adam",
            optimizer_hparams={
                "lr": cfg.learning_rate,
                "weight_decay": 1e-4
            },
            epochs=cfg.epochs,
            log_wandb=cfg.log_wandb,
            train_loader=train_loader,
            test_loader=test_loader,
            include_model_loss=True,
            visualize=cfg.visualize,
            norm_scaler=cfg.norm_scaler,
            entropy_scaler=cfg.entropy_scaler,
            lr_reps=cfg.learning_rate_reps,
        )


