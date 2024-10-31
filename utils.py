import torch
import os
import torchvision
import numpy as np

from datasets import RotatedMNIST, ScaledMNIST

from yacs.config import CfgNode as CN

from models import WSCNN, WSResNet, CNN, ResNet
from gconv import GroupEquivariantCNN, GResNet


def update_config_from_args(cfg, args_dict):
    for key, value in args_dict.items():
        if value is not None:
            keys = key.split('.')
            sub_cfg = cfg
            for sub_key in keys[:-1]:
                sub_cfg = sub_cfg[sub_key]
            sub_cfg[keys[-1]] = value

def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode into a dictionary.
    """
    cfg_dict = {}
    for k, v in cfg_node.items():
        if isinstance(v, CN):
            cfg_dict[k] = cfg_to_dict(v)  # Recurse into nested CfgNode
        else:
            cfg_dict[k] = v
    return cfg_dict

def freeze_representations(model):
    model.first_conv.representations.requires_grad = False

    for conv in model.convs:
        conv.representations.requires_grad = False

def get_model(in_channels, out_channels, cfg, device):
    if cfg.model.lower() == "cnn":
        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": cfg.kernel_size,
            "num_hidden": cfg.num_layers,
            "hidden_channels": cfg.hidden_channels,
            "device":device,
        }
        model = CNN(**hparams)


    elif cfg.model == "GCNN":
        hparams = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": cfg.kernel_size,
                "num_hidden": cfg.num_layers,
                "hidden_channels": cfg.hidden_channels,
                "group_size":cfg.n_group
            }


        model = GroupEquivariantCNN(**hparams)

    elif cfg.model == "WSCNN":
        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": cfg.kernel_size,
            "num_hidden": cfg.num_layers,
            "group_size": cfg.n_group,
            "n_iter": cfg.n_iter,
            "hidden_channels":cfg.hidden_channels,
            "device": device,
            "init_mode": cfg.init_mode,
            "fix_identity": cfg.fix_identity,
        }

        model = WSCNN(**hparams)
    

    elif cfg.model == "GResNet":

        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "hidden_channels": cfg.hidden_channels,
            "kernel_size": cfg.kernel_size,
            "group": cfg.n_group
        }

        model = GResNet(**hparams)

        model.get_parameter_counts()


    elif cfg.model == "ResNet":

        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "hidden_channels": cfg.hidden_channels,
            "kernel_size": cfg.kernel_size
        }

        model = ResNet(**hparams)
        model.get_parameter_counts()

    elif cfg.model == "WSResNet":
        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "group_size": cfg.n_group,
            "kernel_size": cfg.kernel_size,
            "n_iter": cfg.n_iter,
            "hidden_channels":cfg.hidden_channels,
            "init_mode": cfg.init_mode,
            "fix_identity": cfg.fix_identity,
        }

        model = WSResNet(**hparams)

        model.get_parameter_counts()

    return model, hparams



def get_datasets(cfg):
    if "mnist" in cfg.dataset.lower():
        in_channels = 1
        out_channels = 10

    elif cfg.dataset.lower() == "cifar10":
        in_channels = 3
        out_channels = 10

    elif "cifar10" in cfg.dataset.lower(): 
        in_channels = 3
        out_channels = 10

    
    if cfg.dataset.lower() == "rotatedmnist":
        if cfg.group is None:
            angles = [0.0]
        elif "so(2)" in cfg.group.lower():
            angles = None #no subset, use all angles
        elif "c" in cfg.group.lower():
            order = int(cfg.group[1:])
            angles = list(np.linspace(0, 360 - (360 / order), order))
        elif "partial" in cfg.group.lower():
            angles = int(cfg.group.lower().split("_")[1])
        else:
            assert False, f"'{cfg.group}' not recognized."

        if "flip" in cfg.group:
            flip = True
        else:
            flip = False

        if cfg.augment:
            dataset_path = "mnist_dataset.pt"
        else:
            dataset_path = "rotated_mnist_dataset.pt"

        train_dataset = RotatedMNIST(
            root="./data",
            download=True,
            noise_level=0.01, 
            angles=angles,
            transform_labels="all",
            augment=cfg.augment,
            flip=flip,
            dataset_path=dataset_path
        )

        test_dataset = RotatedMNIST(
            root="./data",
            download=True,
            noise_level=0.01, 
            angles=angles,
            train=False,
            flip=flip,
            augment=False
        )
    elif cfg.dataset.lower() == "scaledmnist":
        train_dataset = ScaledMNIST(
            root="./data",
            download=True,
            scales=[0.3, 1.0],
        )

        test_dataset = ScaledMNIST(
            root="./data",
            download=True,
            scales=[0.3, 1.0],
            train=False,
        )
    
    

    elif cfg.dataset.lower() == "cifar10":
        


        if cfg.augment:
            if "partial" in cfg.group.lower():
                angle = int(cfg.group.lower().split("_")[1])
            else:
                angle = 360

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        train_transform = [
            # transforms.RandomCrop(32, padding=6),
            # transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
        test_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
        
    
        if cfg.augment:
            if cfg.group == "flip":
                tform = torchvision.transforms.RandomHorizontalFlip()
            else:
                tform =  torchvision.transforms.RandomRotation(
                    [0, angle],
                    torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0
                )

            train_transform.insert(1, tform)
            test_transform.insert(1, tform)

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train=True, 
            transform=torchvision.transforms.Compose(
                train_transform
            ),
            download=True
        )


        if cfg.augment: #test on single transformation set
            if not os.path.exists(os.path.join("./data", cfg.group + '_cifar10_test.pt')):
                test_dataset = torchvision.datasets.CIFAR10(
                    root="./data", 
                    train=False, 
                    transform=torchvision.transforms.Compose(
                        test_transform
                    ),
                    download=True
                )
                save_transformed_dataset(test_dataset, os.path.join("./data", cfg.group + '_cifar10_test.pt'))
            else:
                # Load the data
                images, labels = torch.load(os.path.join("./data", cfg.group + '_cifar10_test.pt'))
                test_dataset = [*zip(images, labels)]
                
            
        else:
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", 
                train=False, 
                transform=torchvision.transforms.Compose(
                    test_transform
                ),
                download=True
            )

    else:
        assert False, f"dataset '{cfg.dataset}' not recognized."

    return train_dataset, test_dataset, in_channels, out_channels




def save_transformed_dataset(dataset, name=""):
    # Ensure the directory exists
    # os.makedirs(save_dir, exist_ok=False)
    
    transformed_images = []
    labels = []
    
    # Transform and collect all images and labels
    for image, label in dataset:
        transformed_images.append(image)
        labels.append(label)
    
    # Convert lists to tensors
    transformed_images = torch.stack(transformed_images)
    labels = torch.tensor(labels)
    
    # Save the tensors to a file
    torch.save((transformed_images, labels), name)