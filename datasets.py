import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm


class RotatedMNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        noise_level=0.0, 
        angles=None,
        transform_labels="all",
        dataset_path="rotated_mnist_dataset.pt",
        augment=False,
        flip=False,
        **kwargs
    ):
        """
        Rotated MNIST data module: generates train and test sets with samples rotated from [0, 360] degrees. 
        If angles are provided, will instead sample rotations from those.

        Args:
            noise_level (float): variance of gaussian noise applied to each sample after transformation.

            angles (list or None): a list of angles to use for rotations. If None,
                random rotations are used from [0, 360] degrees. providing angles=[0.0]
                recovers regular MNIST.
            transform_labels ("all", None or list): which labels to transform.
            **kwargs: Additional keyword arguments that are passed to the base MNIST
                class constructor. 
        """
        # Initialize the MNIST dataset from torchvision
        super(RotatedMNIST, self).__init__(**kwargs)


        if angles is None:
            angle = 360
        else:
            angle = angles

        self.dataset_path = os.path.join(self.root, f"{'train' if self.train else 'test'}_" + f"{transform_labels}_" + f"{str(angles)}{'_flip_' if flip else ''}" + dataset_path)

        self.noise_level = noise_level
        self.flip = flip
        self.augment = augment
 
        aug_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(
                [0, angle],
                torchvision.transforms.InterpolationMode.BILINEAR,
                fill=0
            ),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]

        if flip:
            aug_transform.insert(1, torchvision.transforms.RandomHorizontalFlip())

        self.aug_transform = torchvision.transforms.Compose(aug_transform)

    
        self.base_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        if transform_labels is None:
            self.transform_labels = []
        elif str(transform_labels).lower() == "all":
            self.transform_labels = list(range(0, 10))
        else:
            self.transform_labels = transform_labels

        if not self.augment:
            if os.path.exists(self.dataset_path):
                self.data, self.labels = torch.load(self.dataset_path)
            else:
                self.data, self.labels = self.create_dataset()
                torch.save((self.data, self.labels), self.dataset_path)

    def create_dataset(self):

        imgs = []
        labels = []

        for idx in tqdm(range(super(RotatedMNIST, self).__len__()), desc="Processing data..."):
            img, target = super(RotatedMNIST, self).__getitem__(idx)
            if target in self.transform_labels:    
                
                rot_img = self.aug_transform(img).clone()
            else:
                rot_img = self.base_transform(img).clone()
            
            rot_img += torch.randn(rot_img.shape) * self.noise_level
            
            imgs.append(rot_img)
            labels.append(target)
        
        return imgs, labels
    
    def __getitem__(self, idx):
        if self.augment:
            img, target = super(RotatedMNIST, self).__getitem__(idx)
            if target in self.transform_labels:
                
                return self.aug_transform(img), target
            else:
                return self.base_transform(img), target
        else:
            return self.data[idx], self.labels[idx]
        


class ScaledMNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        noise_level=0.0, 
        scales=[0.2, 1.0],
        dataset_path="scaled_mnist_dataset.pt",
        **kwargs
    ):
        """
        Rotated MNIST data module: generates train and test sets with samples rotated from [0, 360] degrees. 
        If angles are provided, will instead sample rotations from those.

        Args:
            noise_level (float): variance of gaussian noise applied to each sample after transformation.

            angles (list or None): a list of angles to use for rotations. If None,
                random rotations are used from [0, 360] degrees. providing angles=[0.0]
                recovers regular MNIST.
            transform_labels ("all", None or list): which labels to transform.
            **kwargs: Additional keyword arguments that are passed to the base MNIST
                class constructor. 
        """
        # Initialize the MNIST dataset from torchvision
        super(ScaledMNIST, self).__init__(**kwargs)


        self.scales = scales
        self.dataset_path = os.path.join(self.root, f"{'train' if self.train else 'test'}_" +  str(self.scales) + "_" + dataset_path)

        self.noise_level = noise_level
    
        self.base_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        self.post_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        if not self.train:
            if os.path.exists(self.dataset_path):
                self.data, self.labels = torch.load(self.dataset_path)
            else:
                self.data, self.labels = self.create_dataset()
                torch.save((self.data, self.labels), self.dataset_path)


    def rescale_and_center_image(self, img_tensor, scale_factor):
        """
        Rescale an image tensor and center it within the original dimensions.
        
        Args:
        img_tensor (torch.Tensor): The image tensor to rescale, expected shape [C, H, W]
        scale_factor (float): The factor by which to scale the image
        original_size (tuple): The original dimensions of the image (height, width)
        
        Returns:
        torch.Tensor: The centered image tensor with the same dimensions as the input
        """
        original_size = img_tensor.shape[-2:]
        # Calculate the new size after scaling
        new_height = int(original_size[0] * scale_factor)
        new_width = int(original_size[1] * scale_factor)


        # Resize the image
        resized_img = torchvision.transforms.functional.resize(img_tensor, size=(new_height, new_width))
        
        # Create a new tensor for the output with the same size as the original
        output_tensor = torch.zeros_like(img_tensor)

        # Calculate the padding sizes
        padding_top = (original_size[0] - new_height) // 2
        padding_left = (original_size[1] - new_width) // 2

        # Place the resized image in the center of the original-sized output tensor
        output_tensor[:, padding_top:padding_top + new_height, padding_left:padding_left + new_width] = resized_img[0]

        return output_tensor

    def create_dataset(self):

        imgs = []
        labels = []

        for idx in tqdm(range(super(ScaledMNIST, self).__len__()), desc="Processing data..."):
            img, target = super(ScaledMNIST, self).__getitem__(idx)
                
            factor = np.random.uniform(low=self.scales[0], high=self.scales[1])
            scaled_img = self.post_transform(self.rescale_and_center_image(self.base_transform(img), scale_factor=factor))

            
            imgs.append(scaled_img)
            labels.append(target)
        
        return imgs, labels
    
    def __getitem__(self, idx):
        if self.train:
            img, target = super(ScaledMNIST, self).__getitem__(idx)
            factor = np.random.uniform(low=self.scales[0], high=self.scales[1])
            scaled_img = self.rescale_and_center_image(self.base_transform(img), scale_factor=factor)
            
            return self.post_transform(scaled_img), target
        else:
            return self.data[idx], self.labels[idx]
