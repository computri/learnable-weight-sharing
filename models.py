

import matplotlib.pyplot as plt

import wandb

import torch
import torch.nn as nn

import torch.nn.functional as F


from wsmodules import WSLiftConv2d, WSConv2d



class CNN(nn.Module):
    def __init__(
            self, 
            in_channels=1,
            out_channels=10, 
            kernel_size=5, 
            num_hidden=4, 
            hidden_channels=32, 
            device="cuda",
        ):
        super().__init__()

        
        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=0,
            device=device
        )

        self.convs = nn.ModuleList()
        for i in range(num_hidden):
            self.convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=0,
                    device=device
                )
            )

        self.final_linear = torch.nn.Linear(hidden_channels, out_channels, device=device)

    def forward(self, x):

        x = self.first_conv(x)
        x = F.layer_norm(x, x.shape[-3:])
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = F.layer_norm(x, x.shape[-3:])
            x = F.relu(x)

        # Apply average pooling over remaining spatial dimensions.
        x = F.adaptive_avg_pool2d(x, 1).squeeze()

        x = self.final_linear(x)
        return x

    def visualize(self, logger=None):
        pass


                

    
class WSCNN(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=10, 
        kernel_size=5, 
        num_hidden=4, 
        group_size=4,
        hidden_channels=16, 
        device="cuda",
        n_iter=20,
        init_mode="rand",
        fix_identity=False
    ):
        super().__init__()

        self.first_conv = WSLiftConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            group_size=group_size,
            device=device,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
        )

        self.num_hidden = num_hidden

        self.convs = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.convs.append(
                WSConv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    g_in=group_size,
                    g_out=group_size,
                    device=device,
                    n_iter=n_iter,
                    init_mode=init_mode,
                    fix_identity=fix_identity
                )
            )


        self.group_size = group_size 
        self.hidden_channels = hidden_channels

        self.projection_layer = torch.nn.AdaptiveAvgPool3d(1)
        self.final_linear = torch.nn.Linear(hidden_channels, out_channels, device=device)


    def forward(self, x):

        x = self.first_conv(x)
        x = nn.functional.layer_norm(x, x.shape[-4:])
        x = nn.functional.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = nn.functional.layer_norm(x, x.shape[-4:])
            x = nn.functional.relu(x)
        
        x = self.projection_layer(x).squeeze()
        x = self.final_linear(x)
        return x

    def get_loss(self):
        norm_loss, ent_loss = self.first_conv.get_loss()
        for conv in self.convs:
            n, e = conv.get_loss()
            ent_loss += e 
            norm_loss += n 
        
        return norm_loss, ent_loss
    
    def visualize(self, logger=None):
        """

        """

        rep = self.first_conv.get_representations()
        
        kernel_stack = self.first_conv.get_W().unflatten(0, (self.hidden_channels, self.group_size))

        fig2, ax2 = plt.subplots(1, rep.shape[0], figsize=(rep.shape[0] * 2, 2))
        
        fig3, ax3 = plt.subplots(kernel_stack.shape[1], kernel_stack.shape[0], figsize=(kernel_stack.shape[0] * 3, kernel_stack.shape[1] * 3))

        rep = rep.detach().cpu().numpy()
        kernel_stack = kernel_stack.detach().cpu().numpy()

        n_samples = rep.shape[0]


        for o in range(kernel_stack.shape[0]):
            for g in range(kernel_stack.shape[1]):
                ax3[g, o].imshow(kernel_stack[o, g, 0, :, :])
                ax3[g, o].set_xticks([])
                ax3[g, o].set_yticks([])
        fig3.tight_layout()

        for i in range(n_samples):
            
            ax2[i].imshow(rep[i])

            ax2[i].set_xticks([])
            ax2[i].set_yticks([])


        if logger is not None:
            try: #extreme hacks
                            
                logger.experiment.log({"representations": wandb.Image(fig2)})
                logger.experiment.log({"kernels": wandb.Image(fig3)})
                
            except:
                pass
                
        plt.clf()
        plt.close("all")
    

    

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



    def save_representations(self, name=""):
        print(f"./representations/{name}_rep_0.pt")
        torch.save(self.first_conv.get_representations().detach().cpu(), f"./representations/{name}_rep_0.pt")

        for idx, conv in enumerate(self.convs):
            torch.save(conv.get_representations().detach().cpu(), f"./representations/{name}_rep_{idx+1}.pt")


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )

        # Norm layers:
        self.norm_out = torch.nn.BatchNorm2d(out_channels)

        # Pool
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

        self.norm1 = torch.nn.InstanceNorm2d(out_channels)

        self.norm2 = torch.nn.InstanceNorm2d(out_channels)

        self.out_channels = out_channels
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = torch.nn.functional.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = torch.nn.functional.relu(out)

        if self.shortcut is not None:
            shortcut = x.transpose(1, -1)
            shortcut = self.shortcut(shortcut)
            shortcut = shortcut.transpose(1, -1)
        else:
            shortcut = x
        out = out + shortcut 

        out = self.norm_out(out)
        out = self.pool(out)
        out = torch.nn.functional.relu(out)
        return out



class ResNet(torch.nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        out_channels=10, 
        kernel_size=7, 
        hidden_channels=32
    ):
        super().__init__()
        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding="same"
        )

        self.norm1 = torch.nn.BatchNorm2d(hidden_channels)

        self.resblock1 = ResBlock(
            in_channels = hidden_channels, 
            out_channels = hidden_channels, 
            kernel_size = kernel_size
        )

        self.resblock2 = ResBlock(
            in_channels = hidden_channels, 
            out_channels = 2 * hidden_channels, 
            kernel_size = kernel_size
        )

        self.projection_layer = torch.nn.AdaptiveAvgPool2d(1)
        
        # final linear layer to map from last hidden layer to output
        self.final_linear = torch.nn.Sequential( 
            torch.nn.Linear(2 * hidden_channels, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, out_channels)
        )

    def forward(self, x):

        out = self.first_conv(x)
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
        

        

class WSResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group_size,
        kernel_size=7,
        n_iter=20,
        init_mode="rand",
        fix_identity=False
    ):
        super().__init__()

        self.gconv1 = WSConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            g_in=group_size,
            g_out=group_size,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
        )

        self.gconv2 = WSConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            g_in=group_size,
            g_out=group_size,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
        )

        self.group_size = group_size

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



class WSResNet(torch.nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=10, 
        kernel_size=5, 
        hidden_channels=16, 
        group_size=4,
        n_iter=20,
        init_mode="rand",
        fix_identity=False
    ):
        super().__init__()
        self.lifting_conv = WSLiftConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding="same",
            group_size=group_size,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
        )

        self.norm1 = torch.nn.BatchNorm3d(hidden_channels)

        self.resblock1 = WSResBlock(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            kernel_size=kernel_size,
            group_size=group_size,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
        )

        self.resblock2 = WSResBlock(
            in_channels=hidden_channels, 
            out_channels=2 * hidden_channels, 
            kernel_size=kernel_size,
            group_size=group_size,
            n_iter=n_iter,
            init_mode=init_mode,
            fix_identity=fix_identity
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

    def get_loss(self):
        norm_loss, entropy_loss = self.lifting_conv.get_loss()

        n, e = self.resblock1.gconv1.get_loss()

        norm_loss += n 
        entropy_loss += e
        n, e = self.resblock1.gconv2.get_loss()

        norm_loss += n 
        entropy_loss += e

        n, e = self.resblock2.gconv1.get_loss()

        norm_loss += n 
        entropy_loss += e

        n, e = self.resblock2.gconv2.get_loss()

        norm_loss += n 
        entropy_loss += e

        return norm_loss, entropy_loss


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
        

    def save_representations(self, name=""):
        torch.save(self.lifting_conv.get_representations().detach().cpu(), f"./representations/{name}_rep_0.pt")

        torch.save(self.resblock1.gconv1.get_representations().detach().cpu(), f"./representations/{name}_rep_block1_g1.pt")
        torch.save(self.resblock1.gconv2.get_representations().detach().cpu(), f"./representations/{name}_rep_block1_g2.pt")
        torch.save(self.resblock2.gconv1.get_representations().detach().cpu(), f"./representations/{name}_rep_block2_g1.pt")
        torch.save(self.resblock2.gconv2.get_representations().detach().cpu(), f"./representations/{name}_rep_block2_g2.pt")