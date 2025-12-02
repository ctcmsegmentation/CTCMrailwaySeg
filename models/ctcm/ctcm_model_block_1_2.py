
import torch
import torch.nn as nn
import torch.nn.functional as F


class ctcm_model_block_1_2(torch.nn.Module):
    def __init__(self,
                 block1_weights="block1_weights.pth", 
                 block2_weights="block2_weights.pth"
                ):
        super(ctcm_model_block_1_2, self).__init__()

        self._block2_graph = self._get_normalized_graph()
        self._block1, self._block2 = self._load_models(block1_weights, block2_weights)

    def _get_normalized_graph(self):
        """
        Constructs and returns a normalized adjacency matrix for a graph with nodes arranged in a grid layout with skip-connections.
        
        The method performs the following steps:
        1. Defines the height and width of the grid and calculates the total number of nodes.
        2. Initializes an adjacency matrix with zeros.
        3. Fills the adjacency matrix to create connections between (own) neighboring nodes (up, down, left, right).
        4. Adds connections for nodes with specified diagonal and extended neighbors based on predefined shifts (non-own neighbors).
        5. Adds self-loops to the adjacency matrix.
        6. Computes the degree matrix and its inverse square root.
        7. Normalizes the adjacency matrix using the inverse square root degree matrix.
        
        Returns:
            torch.Tensor: The normalized adjacency matrix of the graph.
        """
        height = 480 // 10
        width = 990 // 10

        num_nodes = height * width
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
        indices = torch.arange(num_nodes).reshape(height, width)

        adjacency_matrix[indices[1:].flatten(), indices[:-1].flatten()] = 1
        adjacency_matrix[indices[:-1].flatten(), indices[1:].flatten()] = 1
        adjacency_matrix[indices[:, 1:].flatten(), indices[:, :-1].flatten()] = 1
        adjacency_matrix[indices[:, :-1].flatten(), indices[:, 1:].flatten()] = 1

        shifts = [
            (2, 0), (1, 1), (1, -1),
            (2, 1), (2, -1), (1, 2), (1, -2)
        ]

        for di, dj in shifts:
            valid_i = torch.arange(height - di)
            valid_j = torch.arange(width)
            if dj != 0:
                valid_j = valid_j[(valid_j + dj >= 0) & (valid_j + dj < width)]

            src_indices = indices[valid_i[:, None], valid_j].flatten()
            dest_indices = indices[(valid_i + di)[:, None], valid_j + dj].flatten()
            adjacency_matrix[src_indices, dest_indices] = 0.1   

        adjacency_matrix += (torch.eye(adjacency_matrix.shape[0])).float()
        D_hat = torch.sum(adjacency_matrix, dim=1).diag()

        eigenvalues, eigenvectors = torch.linalg.eigh(D_hat)
        inv_sqrt_eigenvalues = torch.diag(torch.pow(eigenvalues, -0.5))
        D_sqrt_inv = torch.matmul(torch.matmul(eigenvectors, inv_sqrt_eigenvalues), eigenvectors.T)

        normalized_adjacency = torch.matmul(torch.matmul(D_sqrt_inv, adjacency_matrix), D_sqrt_inv).float()
        return normalized_adjacency

    def _load_models(self, block1_weights, block2_weights):
        """
        Load pretrained weights into models: block1 and block2.

        Args:
            block1_weights (str): File path to the pretrained weights for block1.
            block2_weights (str): File path to the pretrained weights for block2.

        Returns:
            tuple: A tuple containing three initialized and loaded models (block1, block2).
        """
        block1 = UNet(3, 2)
        block1.load_state_dict(torch.load(block1_weights, map_location='cpu', weights_only=True))
        block1.eval()

        block2 = GCNmodel(1, 2, self._block2_graph)
        block2.load_state_dict(torch.load(block2_weights, map_location='cpu', weights_only=True))
        block2.eval()

        return block1, block2

    def forward(self, x):
        """
        Forward pass method for processing the input tensor `x` through the first and second blocks of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            block2_out_postprocess (torch.Tensor): Processed output tensor from Block 2.
        """
        
        x = x.to(torch.float32)

        # Resize input tensor for block 1
        block1_in = F.interpolate(
            x, size=(512, 1024), 
            mode='bilinear', antialias=True
        )

        # Pass through block 1 and obtain a binary segementation mask denoting railyard
        block1_out = torch.argmax(
            self._block1(block1_in), dim=1
        ).to(torch.float32)
        
        # Resize segmentation mask for input to block 2 and perform average-2/3-pooling
        block2_in_mask = F.interpolate(
            block1_out.unsqueeze(0), size=(480, 990), 
            mode='bilinear', antialias=True
        ).squeeze(0)
        block2_in_mask = (F.avg_pool2d(block2_in_mask, (10, 10)) >= 0.6).float()
        block2_in_mask = torch.flatten(block2_in_mask, start_dim=1).unsqueeze(-1)

        # Resize input image for the block 2 and perform average pooling
        block2_in_image = F.interpolate(
            block1_in, size=(480, 990), 
            mode='bilinear', antialias=True
        )
        block2_in_image = F.avg_pool2d(block2_in_image, (10, 10))
        block2_in_image = torch.flatten(block2_in_image, start_dim=2)
        block2_in_image = torch.permute(block2_in_image, (0, 2, 1))

        # Pass through block 2 and obtain processed binary segmentation matrix denoting postprocessed railyard segments
        block2_out = torch.argmax(
            self._block2(block2_in_mask, block2_in_image), dim=2
        ).to(torch.float32)

        block2_out_postprocess = torch.unflatten(torch.round(block2_out).to(torch.long), 1, (48, 99))
        block2_out_postprocess = torch.clamp(
            F.conv2d(
                block2_out_postprocess.unsqueeze(1).float(), 
                torch.ones((1, 1, 3, 3)).to(block2_out_postprocess.device), 
                padding=(1, 1)
            ), 0, 1
        ).squeeze(1).to(torch.float32)
        
        
        block2_out_postprocess = block2_out_postprocess.repeat_interleave(10, dim=1).repeat_interleave(10, dim=2)
        block2_out_postprocess = F.interpolate(
            block2_out_postprocess.unsqueeze(0), size=(512, 1024), 
            mode='nearest'
        ).squeeze(0)

        return block2_out_postprocess
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        unit = 4

        self.inc = (DoubleConv(n_channels, 2*unit))
        self.down1 = (Down(2*unit, 4*unit))
        self.down2 = (Down(4*unit, 8*unit))
        self.down3 = (Down(8*unit, 16*unit))
        factor = 2 if bilinear else 1
        self.down4 = (Down(16*unit, 32*unit // factor))
        self.up1 = (Up(32*unit, 16*unit // factor, bilinear))
        self.up2 = (Up(16*unit, 8*unit // factor, bilinear))
        self.up3 = (Up(8*unit, 4*unit // factor, bilinear))
        self.up4 = (Up(4*unit, 2*unit, bilinear))
        self.outc = (OutConv(2*unit, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


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


class GCNmodel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, adj):
        super().__init__()
        self.conv0_mask = GraphConvolution(num_node_features, 3, adj)
        self.conv1_mask = GraphConvolution(3, 6, adj)
        self.conv2_mask = GraphConvolution(6, 12, adj)
        self.conv3_mask = GraphConvolution(12, 18, adj)

        self.conv0_image = GraphConvolution(3, 6, adj)
        self.conv1_image = GraphConvolution(6, 12, adj)
        self.conv2_image = GraphConvolution(12, 18, adj)
        self.conv3_image = GraphConvolution(18, 18, adj)

        self.conv1 = GraphConvolution(2*18, 18, adj)
        self.conv2 = GraphConvolution(18, 18, adj)
        self.conv3 = GraphConvolution(18, 6, adj)
        self.conv4 = GraphConvolution(6, num_classes, adj)

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params

    def forward(self, mask, img):
        x_in_mask = mask
        x_in_image = img

        x0_image = self.conv0_image(x_in_image)
        x0_mask = self.conv0_mask(x_in_mask)

        x1_image = self.conv1_image(x0_image)
        x1_mask = self.conv1_mask(x0_mask)
        
        x2_image = self.conv2_image(x1_image)
        x2_mask = self.conv2_mask(x1_mask)

        x3_image = self.conv3_image(x2_image)
        x3_mask = self.conv3_mask(x2_mask)

        x4 = torch.cat((x3_mask, x3_image), dim=2)

        x5 = self.conv1(x4)
        x6 = self.conv2(x5)
        x7 = self.conv3(x6)
        x8 = self.conv4(x7)
            
        return x8
    

class GraphConvolution(torch.nn.Module):
    def __init__(self, input_dim, output_dim, adj):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Linear(input_dim, output_dim)
        self.adj = adj

    def forward(self, X):
        """
        X: Node feature matrix (N x input_dim)
        """
        tmp = self.weights(X)
        output = torch.empty_like(tmp)
        self.adj = self.adj.to(X.device)
        for i in range(tmp.size(0)):
            output[i, :, :] = torch.mm(self.adj, tmp[i, :, :])
        return output
    