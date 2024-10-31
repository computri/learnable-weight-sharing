import torch
from se2cnn.nn.conv import R2ToSE2Conv, SE2ToSE2Conv, SE2ToR2Conv, SE2ToR2Projection


class GroupEquivariantCNN(torch.nn.Module):

    def __init__(
            self, 
            group_size, 
            in_channels, 
            out_channels, 
            kernel_size, 
            num_hidden, 
            hidden_channels
            ):
        super().__init__()


        # Create the lifing convolution.

        self.lifting_conv = R2ToSE2Conv(
            input_dim=in_channels, 
            output_dim=hidden_channels, 
            num_theta=group_size, 
            kernel_size=kernel_size, 
            padding=0, 
            stride=1,
            bias=True, 
            diskMask=True, 
            groups=1
        )
        

        # Create a set of group convolutions.
        self.gconvs = torch.nn.ModuleList()

        for i in range(num_hidden):
            self.gconvs.append(
                SE2ToSE2Conv(
                    input_dim=hidden_channels, 
                    output_dim=hidden_channels, 
                    num_theta=group_size, 
                    kernel_size=kernel_size, 
                    padding=0, 
                    stride=1, 
                    bias=True, 
                    diskMask=True, 
                    groups=1
                )
            )

        # Create the projection layer. Hint: check the import at the top of
        # this cell.
        
        self.projection_layer = SE2ToR2Projection("mean")

        self.spatial_projection_layer = torch.nn.AdaptiveAvgPool2d(1)

        # And a final linear layer for classification.
        self.final_linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        
        # Lift and disentangle features in the input.
        x = self.lifting_conv(x)
        x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)

        # Apply group convolutions.
        for gconv in self.gconvs:
            x = gconv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)
        
        # to ensure equivariance, apply max pooling over group and spatial dims.
        x = self.projection_layer(x)

        x = self.spatial_projection_layer(x).squeeze()

        x = self.final_linear(x)
        return x
    
    def visualize(self, logger=None):
        pass

class GResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group,
    ):
        super().__init__()


        self.gconv1 = SE2ToSE2Conv(
            input_dim=in_channels, 
            output_dim=out_channels, 
            num_theta=group, 
            kernel_size=7, 
            padding="same"
        )

        self.gconv2 = SE2ToSE2Conv(
            input_dim=out_channels,
            output_dim=out_channels,
            kernel_size=7,
            num_theta=group,
            padding="same"
        )


        self.group_size = group

        # Norm layers:
        self.norm_out = torch.nn.BatchNorm3d(out_channels)

        # Pool
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

        self.out_channels = out_channels
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.gconv1(x)
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out = torch.nn.functional.relu(out)

        out = self.gconv2(out)
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out = torch.nn.functional.relu(out)

        if self.shortcut is not None:
            shortcut = x.transpose(1, -1)
            shortcut = self.shortcut(shortcut)
            shortcut = shortcut.transpose(1, -1)
        else:
            shortcut = x
        out = out + shortcut 

        out = self.norm_out(out)
        out = self.pool(out.flatten(1,2)).unflatten(dim=1, sizes=(self.out_channels, self.group_size))
        out = torch.nn.functional.relu(out)
        return out



class GResNet(torch.nn.Module):
    def __init__(
        self, 
        group, 
        in_channels=3, 
        out_channels=10, 
        kernel_size=7, 
        hidden_channels=32
    ):
        super().__init__()


        self.lifting_conv = R2ToSE2Conv(
            input_dim=in_channels, 
            output_dim=hidden_channels, 
            num_theta=group, 
            kernel_size=kernel_size, 
            padding=0, 
            bias=True,
        )

        self.norm1 = torch.nn.BatchNorm3d(hidden_channels)

        self.resblock1 = GResBlock(
            in_channels = hidden_channels, 
            out_channels = hidden_channels, 
            group = group
        )

        self.resblock2 = GResBlock(
            in_channels = hidden_channels, 
            out_channels = 2 * hidden_channels, 
            group = group
        )

        self.projection_layer = torch.nn.AdaptiveAvgPool3d(1)
        
        # final linear layer to map from last hidden layer to output
        self.final_linear = torch.nn.Sequential( 
            torch.nn.Linear(2 * hidden_channels, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, out_channels)
        )

    def forward(self, x):

        out = self.lifting_conv(x)
        out = self.norm1(out)
        out = torch.nn.functional.relu(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.projection_layer(out).squeeze()
        out = self.final_linear(out)

        return out
    
    def visualize(self, logger=None):
        pass

    def get_parameter_counts(self):
        
        representations = 0.0
        rest = 0.0
        
        for p in self.named_parameters():
            if p[1].requires_grad:
                if "representations" in p[0]:
                    representations += p[1].numel()
                else:
                    rest += p[1].numel()

        print(f"# free parameters: {rest}, # representations parameters: {representations}, Total: {rest + representations}")
           

