
import torch
import torch.nn as nn
import torch.nn.functional as F


class ctcm_model_block_3(torch.nn.Module):
    def __init__(self, block3_weights="block3_weights.pth"):
        super(ctcm_model_block_3, self).__init__()
        self._block3 = self._load_models(block3_weights)

    def _load_models(self, block3_weights):
        """
        Load pretrained weights into the block3 model.

        Args:
            block3_weights (str): File path to the pretrained weights for block3.

        Returns:
            loaded model block3.
        """        
        block3 = UNetGnnRbf(3, 2)
        block3.load_state_dict(torch.load(block3_weights, map_location='cpu', weights_only=True), strict=False)
        block3.eval()
        return block3
    
    def forward(self, x, img):
        """
        Forward pass method for processing tensors `x` and `img` through the rails block of the model.

        Args:
            x (torch.Tensor): Tensor with shape (batch, 512, 1024). Denoting railyard segment where the rails segmentation will take place.
            img (torch.Tensor): Tensor with shape (batch, 3, 512, 1024). Input image.

        Returns:
            torch.Tensor: Processed output tensor from the rails block.
        """
        img = torch.where(
            x.unsqueeze(1) == 0, 
            torch.zeros_like(img), 
            img
        )
        block3_out = torch.argmax(self._block3(img), dim=1).float()
        block3_out_postprocess = torch.clamp(
            F.conv2d(
                block3_out.unsqueeze(1).float(), 
                torch.ones((1, 1, 3, 3)).to(block3_out.device), 
                padding=(1, 1)
            ), 0, 1
        ).squeeze(1) 
        return block3_out_postprocess
    

class UNetGnnRbf(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetGnnRbf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dynamic_lambda_scale = 1.0

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        
        factor = 2 if bilinear else 1
        self.down5 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

        self.bottleneck = GNNbottleneck(16, 32, 128)

    def print_model_params_cnt(self):
        print("Model parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        xinc = self.inc(x)
        xd1 = self.down1(xinc)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xd5 = self.down5(xd4)

        x = torch.flatten(xd5, start_dim=2).permute(0, 2, 1)
        x = self.bottleneck(x)
        x = torch.unflatten(x, 1, (16, 32)).permute(0, 3, 1, 2)

        x = x + xd5

        x = self.up1(x, xd4)
        x = self.up2(x, xd3)
        x = self.up3(x, xd2)
        x = self.up4(x, xd1)
        x = self.up5(x, xinc)
        logits = self.outc(x)

        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.outc = torch.utils.checkpoint(self.outc)
        self.bottleneck = torch.utils.checkpoint(self.bottleneck)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, edges):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.ReLU(inplace=True)
        self.edges = edges

    def forward(self, nodes):
        # Ensure edges are on the same device as the nodes
        self.edges = self.edges.to(nodes.device)
        
        B, N, F = nodes.size()
        src, dest = self.edges[:, 0], self.edges[:, 1]

        # Initialize aggregated neighbors tensor
        aggregated_neighbors = torch.zeros((B, N, F), dtype=nodes.dtype, device=nodes.device)

        # Scatter add to aggregate neighbor features
        aggregated_neighbors = aggregated_neighbors.scatter_add_(
            1, dest.unsqueeze(0).unsqueeze(-1).expand(B, -1, F), nodes[:, src]
        )

        # Compute degree of each node
        degree = torch.zeros((B, N, 1), dtype=nodes.dtype, device=nodes.device).scatter_add_(
            1, dest.unsqueeze(0).unsqueeze(-1), torch.ones((B, dest.shape[0], 1), device=nodes.device)
        )

        # Normalize neighbors by degree
        degree = degree + 1  # Add 1 to avoid division by zero
        normalized_neighbors = aggregated_neighbors / degree

        # Apply weight and bias
        out = normalized_neighbors @ self.weight + self.bias

        # Apply activation function
        return self.activation(out)


class GNNbottleneck(nn.Module):
    def __init__(self, height, width, features):
        super(GNNbottleneck, self).__init__()
        self.edges = self._get_normalized_graph(height, width)
        self.gcn1 = GraphConvolutionLayer(features, features//2, self.edges)
        self.gcn2 = GraphConvolutionLayer(features//2, features, self.edges)

    def forward(self, x):
        return self.gcn2(self.gcn1(x))
    
    def _get_normalized_graph(self, height, width):
        shifts = [
            (2, 0), (1, 1), (1, -1),
            (2, 1), (2, -1), (1, 2), (1, -2)
        ]
        edges = []
        for y in range(height):
            for x in range(width):
                node = y * width + x
                for dy, dx in shifts:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor = ny * width + nx
                        edges.append((node, neighbor))
        return torch.tensor(edges, dtype=torch.long)