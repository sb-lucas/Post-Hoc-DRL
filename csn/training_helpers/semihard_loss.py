import torch
import torch.nn as nn
import numpy as np

class SemiHardTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, category_binary_tags, artist_ids):
        """embeddings: (B, D) -> Model output embeddings for batch size B
        category_binary_tags: (B, T) -> One-hot binary matrix for tags in a
        dimension (e.g. Genre)
        """
        B = embeddings.size(0)

        # Pairwise L2 Distances Matrix (B, B)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Shared tags matrix (B, B)
        shared_tags = torch.matmul(category_binary_tags, category_binary_tags.T)

        # Positive Mask: Shares >=1 tag, not self
        pos_mask = (shared_tags > 0)
        pos_mask.fill_diagonal_(False)

        # 2D Artist Inequality Matrix (B, B)
        diff_artist_2d = artist_ids.unsqueeze(1) != artist_ids.unsqueeze(0)
        
        # Base Negative Criteria: Shares 0 tags with Anchor AND has >=1 tag in category
        has_any_tag = (category_binary_tags.sum(dim=1) > 0).unsqueeze(0)  # (1, B)
        neg_mask_base = (shared_tags == 0) & has_any_tag 
        neg_mask_base.fill_diagonal_(False)

        # Broadcast to 3D Triplet Masks (B, B, B) -> [Anchor, Positive, Negative]
        d_ap = dist_matrix.unsqueeze(2).expand(-1, -1, B)  # (B, B, B)
        d_an = dist_matrix.unsqueeze(1).expand(-1, B, -1)  # (B, B, B)

        # Valid triplet mask combining positive and negative criteria
        valid_triplet_mask = pos_mask.unsqueeze(2) & neg_mask_base.unsqueeze(1)

        # Enforce N != A AND N != P across the 3D triplet space
        diff_artist_N_A = diff_artist_2d.unsqueeze(1)  # (B, 1, B) -> (A, P, N)
        diff_artist_N_P = diff_artist_2d.unsqueeze(0)  # (1, B, B) -> (A, P, N)
        artist_neg_mask = diff_artist_N_A & diff_artist_N_P

        # Final Valid Triplet Mask combining tags + artist constraints
        valid_triplet_mask = valid_triplet_mask & artist_neg_mask

        # Semi-hard condition: d(A, P) < d(A, N) < d(A, P) + margin
        semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + self.margin)
        active_mask = valid_triplet_mask & semi_hard_mask

        # Fallback: take any active triplets (loss > 0) if no semi-hard triplets exist
        if active_mask.sum() == 0:
            active_mask = valid_triplet_mask & (d_an < d_ap + self.margin)

        # Return 0 if no valid triplets exist in the batch
        if active_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Extract active distances as 1D vectors
        d_ap_active = d_ap[active_mask]
        d_an_active = d_an[active_mask]
        target = torch.ones_like(d_an_active)

        # Delegate computation to PyTorch's MarginRankingLoss
        return self.ranking_loss(d_an_active, d_ap_active, target)