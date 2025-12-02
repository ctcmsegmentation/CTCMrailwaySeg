
import torch
from torch.nn import functional as F


class CorrectionEnlargingBlocks:
    def __init__(self, device,
                 delay=10, epsilon=1/4, enlarging_tolerance=2/3, 
                ):
        
        self.device = device
        
        self.delay = delay
        self.epsilon = epsilon
        self.enlarging_tolerance = enlarging_tolerance
    

    def correction_policy(self, mask):
        
        _, num_cols = mask.shape
        m = num_cols // 2 if num_cols % 2 == 1 else (num_cols // 2) - 1

        L = mask[:, :m]
        R = mask[:, m + 1:]

        alpha_L = m - torch.argmax(L, dim=1)
        alpha_R = m - torch.argmax(R.flip(dims=[1]), dim=1)

        beta_L = -torch.argmax(L.flip(dims=[1]), dim=1)
        beta_R = -torch.argmax(R, dim=1)

        kappa_L = torch.any(L == 1, dim=1)
        kappa_R = torch.any(R == 1, dim=1)

        gamma_L = torch.where(kappa_L, alpha_L, 
                    torch.where(kappa_R, beta_R, 0)
                ).float()
        gamma_R = torch.where(kappa_R, alpha_R, 
                    torch.where(kappa_L, beta_L, 0)).float()

        delta = torch.mean(torch.abs(gamma_L - gamma_R)).item()

        return delta
    
    def enlarging_policy(self, mask):
        c7_filter = torch.ones((1, 1, 1, 7)).to(self.device)
        c11_filter = torch.ones((1, 1, 11, 1)).to(self.device)
        c5x5_filter = torch.ones((1, 1, 5, 5)).to(self.device)
        c7x7_filter = torch.ones((1, 1, 7, 7)).to(self.device)
        c11x11_filter = torch.ones((1, 1, 11, 11)).to(self.device)

        mask_nonzero_idx = int(mask.nonzero()[0][1])
        limit = int(max(mask_nonzero_idx * 2/3, 11))
        strip = (512 - limit) // 5

        for i in range(5):
            region = mask[:, max(limit + i * strip - 11, 0): min(limit + (i + 1) * strip + 11, 512), :]
            region = F.conv2d(region, c7x7_filter, padding=(3, 3))
            region = torch.where(region < (i+1)*8, 0.0, 1.0)
            region = F.conv2d(region, c7x7_filter, padding=(3, 3))
    
            for _ in range(i + 2):
                region = F.conv2d(region, c7_filter, padding=(0, 3))
            mask[:, max(limit + i * strip - 11, 0): min(limit + (i + 1) * strip + 11, 512), :] = region

        mask = torch.clamp(mask, 0, 1).float()
        region = mask[:, limit + strip * 2:, :]
        region = F.conv2d(region, c11_filter, padding=(5, 0))
        region = torch.where(region >= 8, 1, 0)
        mask[:, limit + strip * 2:, :] = region

        limit = max(mask_nonzero_idx, 11)
        strip = (512 - limit) // 5
        for i in range(5):
            region = mask[:, max(limit + i * strip - 11, 0): min(limit + (i + 1) * strip + 11, 512), :]
            for _ in range(min(i+2, 6)):
                region = F.conv2d(region, c5x5_filter, padding=(2, 2))
            mask[:, max(limit + i * strip - 11, 0): min(limit + (i + 1) * strip + 11, 512), :] = region
        mask = torch.clamp(mask, 0, 1)

        limit = torch.nonzero(mask)[0][1].item()
        mask[:,:limit+1, :] = F.conv2d(mask[:, :limit+1, :], c11x11_filter, padding=5)
        mask[mask > 0] = 1
        return mask
