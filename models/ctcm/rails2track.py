
import torch
import torch.nn as nn


class rails2track(nn.Module):
    def __init__(self, railsLabel=1, trackLabel=2, margin=.2):
        super(rails2track, self).__init__()
        self.margin = margin
        self.railsLabel = railsLabel
        self.trackLabel = trackLabel

    def polynom_fit(self, x, y, degree1:int, degree2:int):
        x = x.float()
        y = y.float()
        x = x / 1024
        y = y / 1024

        X1 = torch.stack([x**i for i in range(degree1 + 1)], dim=1)
        U1, S1, Vh1 = torch.linalg.svd(X1, full_matrices=False)
        S_inv1 = torch.diag(1 / S1)
        coefficients1 = Vh1.T @ S_inv1 @ U1.T @ y

        X2 = torch.stack([x**i for i in range(degree2 + 1)], dim=1)
        U2, S2, Vh2 = torch.linalg.svd(X2, full_matrices=False)
        S_inv2 = torch.diag(1 / S2)
        coefficients2 = Vh2.T @ S_inv2 @ U2.T @ y

        return coefficients1, coefficients2
    
    def polynom_predict(self, x, coefficients1, coefficients2):
        x = x.float()
        x1 = x[x<256]
        x2 = x[x>=256]

        result1 = torch.zeros_like(x1)
        result2 = torch.zeros_like(x2)
        
        if len(x1) > 0:
            x1 = x1 / 1024
            degree1 = coefficients1.shape[0] - 1
            X = torch.stack([x1**i for i in range(degree1 + 1)], dim=1)
            result1 = (X @ coefficients1) * 1024

        if len(x2) > 0:
            x2 = x2 / 1024
            degree2 = coefficients2.shape[0] - 1
            X = torch.stack([x2**i for i in range(degree2 + 1)], dim=1)
            result2 = (X @ coefficients2) * 1024

        return torch.cat((result1, result2))


    def get_cnt_clusters_per_row(self, mask):
        mask_comparison = mask == self.railsLabel
        mask_expanded = mask_comparison.unsqueeze(2)
        transitions = torch.cat([
                torch.zeros(mask_expanded.shape[0], 1, dtype=mask_expanded.dtype), 
                torch.diff(mask_expanded.flatten(start_dim=1), dim=1)
            ], dim=1
        )
        cluster_start = torch.sum(transitions > 0, dim=1)
        return cluster_start//2

    def get_ncluster_rows(self, mask, clusters_per_row, required_cnt:int, gteq:bool=False):
        if gteq:
            idxs = torch.where(clusters_per_row > required_cnt)[0]
        else:
            idxs = torch.where(clusters_per_row == required_cnt)[0]
        return mask[idxs, :], idxs

    def get_first_last_cluster_idxs(self, mask, rows):
        first_indices = torch.argmax(rows, dim=1)
        last_indices = mask.size(1) - torch.argmax(rows.flip(1), dim=1)
        return first_indices, last_indices

    def get_curves(self, mask, rows_idxs, first_indices, last_indices):
        left = torch.stack((rows_idxs, first_indices), dim=1)
        right = torch.stack((rows_idxs, last_indices), dim=1)
        delta = torch.stack((rows_idxs, last_indices - first_indices), dim=1)
        leftBorder = torch.clamp((first_indices - self.margin * delta[:, 1]).int(), min=0, max=mask.shape[1] - 1)
        rightBorder = torch.clamp((last_indices + self.margin * delta[:, 1]).int(), min=0, max=mask.shape[1] - 1)
        return left, right, leftBorder, rightBorder, delta

    def fill_iterleave_railtrack(self, mask, leftBorder, rightBorder, rows_idxs):
        lengths = rightBorder - leftBorder + 1
        arange_tensor = torch.arange(lengths.max()).unsqueeze(0)
        arange_mask = arange_tensor < lengths.unsqueeze(1)
        col_indices = (arange_tensor * arange_mask + leftBorder.unsqueeze(1)).masked_select(arange_mask)
        row_indices = torch.repeat_interleave(rows_idxs, rightBorder - leftBorder + 1)
        mask[row_indices, col_indices] = self.trackLabel
        return mask
    
    def forward(self, mask):
        rails_mask = mask.clone()
        clusters_per_row = self.get_cnt_clusters_per_row(mask)
        
        # two clusters
        two_clusters_rows, two_clusters_rows_idxs = self.get_ncluster_rows(mask, clusters_per_row, 2)
        
        first_idxs, last_idxs = self.get_first_last_cluster_idxs(mask, two_clusters_rows)
        left, right, leftBorder, rightBorder, delta = self.get_curves(
            mask, two_clusters_rows_idxs, first_idxs, last_idxs
        )
        mask = self.fill_iterleave_railtrack(mask, leftBorder, rightBorder, two_clusters_rows_idxs)

        leftPolynomWeights_pol1, leftPolynomWeights_pol2 = self.polynom_fit(left[:, 0], left[:, 1], 8, 2)
        rightPolynomWeights_pol1, rightPolynomWeights_pol2 = self.polynom_fit(right[:, 0], right[:, 1], 8, 2)
        centerPolynomWeights_pol1, centerPolynomWeights_pol2 = self.polynom_fit(right[:, 0], (left[:, 1].float() + right[:, 1].float())/2, 8, 2)
        deltaPolynomWeights_pol1, deltaPolynomWeights_pol2 = self.polynom_fit(delta[:, 0], delta[:, 1], 2, 2) 

        # single cluster
        single_clusters_rows, single_clusters_rows_idxs = self.get_ncluster_rows(mask, clusters_per_row, 1)
        if len(single_clusters_rows_idxs) > 0:
            lefts = self.polynom_predict(single_clusters_rows_idxs, leftPolynomWeights_pol1, leftPolynomWeights_pol2)
            rights = self.polynom_predict(single_clusters_rows_idxs, rightPolynomWeights_pol1, rightPolynomWeights_pol2)
            deltas = torch.abs(self.polynom_predict(single_clusters_rows_idxs, deltaPolynomWeights_pol1, deltaPolynomWeights_pol2))

            idxs = torch.argmax(single_clusters_rows, dim=1)
            condition = self.polynom_predict(single_clusters_rows_idxs, centerPolynomWeights_pol1, centerPolynomWeights_pol2) > idxs
            
            leftBorder = torch.where(
                condition, 
                torch.clamp(idxs - self.margin * deltas, min=0, max=mask.shape[1] - 1),
                torch.clamp(idxs - (1 + self.margin) * deltas, min=0, max=mask.shape[1] - 1)
            ).long()
            rightBorder = torch.where(
                condition, 
                torch.clamp(idxs + (1 + self.margin) * deltas, min=0, max=mask.shape[1] - 1),
                torch.clamp(idxs + self.margin * deltas, min=0, max=mask.shape[1] - 1)
            ).long()

            mask = self.fill_iterleave_railtrack(mask, leftBorder, rightBorder, single_clusters_rows_idxs)

        # more clusters
        _, more_clusters_rows_idxs = self.get_ncluster_rows(mask, clusters_per_row, 2, gteq=True)
        if len(more_clusters_rows_idxs) > 0:
            lefts = self.polynom_predict(more_clusters_rows_idxs, leftPolynomWeights_pol1, leftPolynomWeights_pol2)
            rights = self.polynom_predict(more_clusters_rows_idxs, rightPolynomWeights_pol1, rightPolynomWeights_pol2)
            deltas = torch.abs(self.polynom_predict(more_clusters_rows_idxs, deltaPolynomWeights_pol1, deltaPolynomWeights_pol2))
            centers = (lefts + rights)/2
            
            leftBorder = torch.clamp(centers - (0.5 + self.margin) * deltas, min=0, max=mask.shape[1] - 1).long()
            rightBorder = torch.clamp(centers + (0.5 + self.margin) * deltas, min=0, max=mask.shape[1] - 1).long()

            mask = self.fill_iterleave_railtrack(mask, leftBorder, rightBorder, more_clusters_rows_idxs)
            
        mask[mask==self.railsLabel] = 0
        mask = torch.where((rails_mask == self.railsLabel) & (mask == self.trackLabel), self.railsLabel, mask)
        
        return mask
