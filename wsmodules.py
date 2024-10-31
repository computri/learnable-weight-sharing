import torch


import math
import torch.nn.init as init
import torch

import torch.nn as nn





def is_normalized2d(tensor):
    return torch.allclose(
        tensor.sum(dim=-1), torch.ones(tensor.shape[-1], device=tensor.device)
    ) and torch.allclose(
        tensor.sum(dim=-2), torch.ones(tensor.shape[-2], device=tensor.device)
    )




class WSBase(nn.Module):
    def __init__(
        self,
        n_iter=20,
        init_mode="rand",
        ):
        super().__init__()

        self.n_iter = n_iter

        self.init_mode = init_mode


    def norm_loss(self, tensor):
        loss = torch.mean(
            torch.logsumexp(tensor, -2, keepdim=True) ** 2
            + torch.logsumexp(tensor, -1, keepdim=True) ** 2
        )
        return loss


    def entropy_loss(self, tensor):
        loss = -torch.sum(tensor * torch.clamp(tensor, min=1e-16).log())
        return loss

    def log_sinkhorn(self, log_alpha):
        """Performs incomplete Sinkhorn normalization to log_alpha."""
        i = 0
        while not is_normalized2d(log_alpha.exp()):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
            i += 1
            if i == self.n_iter:
                break
        return log_alpha.exp()


    def get_representations(self):
        representations = self.log_sinkhorn(self.representations)

        if self.fix_identity:
            representations = torch.cat((self.id, representations), dim=0)
        return representations

    def get_loss(self):
        # loss on the representations
        representations = self.get_representations()

        norm_loss = self.norm_loss(self.representations)
        entropy_loss = self.entropy_loss(representations)

        return norm_loss, entropy_loss




class WSLiftConv2d(WSBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_size,
        n_iter,
        device="cuda",
        bias=True,
        init_mode="rand",
        padding=0,
        fix_identity=False

    ):
        super().__init__(
            n_iter=n_iter, 
            init_mode=init_mode, 
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size * kernel_size, device=device
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, 1, 1, 1, device=device))
        else:
            self.register_buffer("bias", None)
        
        self.padding = padding

        self.group_size = group_size

        d_in = kernel_size ** 2

        
        self.fix_identity = fix_identity


        if fix_identity:
            group_size -= 1
            self.id = nn.Parameter(torch.eye(d_in, device=device).unsqueeze(0)+ 1e-16, requires_grad=False)

        if self.init_mode == "rand":
            self.representations = nn.Parameter((torch.rand(group_size, d_in, d_in, device=device)))
        elif self.init_mode == "kaiming_uniform":
            self.representations = nn.Parameter(torch.empty(group_size, d_in, d_in, device=device))
            init.kaiming_uniform_(self.representations)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        W = self.get_W()
        out = torch.nn.functional.conv2d(input, W, padding=self.padding).unflatten(
            1, (self.out_channels, self.group_size)
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def get_W(self):
        rep = self.get_representations()
        kernel = torch.einsum("gnm,oim->ogin", rep, self.weight)
        kernel = kernel.flatten(start_dim=0, end_dim=1).unflatten(
            -1, (self.kernel_size, self.kernel_size)
        )
        return kernel


class WSConv2d(WSBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        g_in,
        g_out,
        n_iter,
        device="cuda",
        bias=True,
        init_mode="rand",
        padding=0,
        fix_identity=False

    ):
        super().__init__(
            n_iter=n_iter,
            init_mode=init_mode,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.g_in = g_in

        self.g_out = g_out

        self.D = kernel_size**2

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels, self.in_channels, self.g_in * self.D, device=device
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1, 1, device=device))
        else:
            self.register_buffer("bias", None)

        self.n_iter = n_iter

        self.padding = padding
        
        d_in = self.g_in * self.D
        group_size = self.g_out

        self.fix_identity = fix_identity
        if fix_identity:
            group_size -= 1
            self.id = nn.Parameter(torch.eye(d_in, device=device).unsqueeze(0)+ 1e-16, requires_grad=False)

        if self.init_mode == "rand":
            self.representations = nn.Parameter((torch.rand(group_size, d_in, d_in, device=device)))
        elif self.init_mode == "kaiming_uniform":
            self.representations = nn.Parameter(torch.empty(group_size, d_in, d_in, device=device))
            init.kaiming_uniform_(self.representations)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        input = input.flatten(start_dim=1, end_dim=2)
        W = self.get_W()
        out = torch.nn.functional.conv2d(input, W, padding=self.padding).unflatten(
            1, (self.out_channels, self.g_out)
        )

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_W(self):
        rep = self.get_representations()
        kernel = torch.einsum("gnm,oim->ogin", rep, self.weight)
        kernel = kernel.flatten(start_dim=0, end_dim=1).unflatten(
            -1, (self.g_in, self.kernel_size, self.kernel_size)
        )
        kernel = kernel.flatten(start_dim=1, end_dim=2)

        return kernel

