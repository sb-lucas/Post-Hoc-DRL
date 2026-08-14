import torch
import torch.nn as nn
import torch.nn.functional as F

class CS_Tripletnet(nn.Module):
    def __init__(self, csn):
        super(CS_Tripletnet, self).__init__()
        self.csn = csn

    def forward(self, anchor, negative, positive, condition):
        emb_anchor, masknorm_norm_anchor = self.csn(anchor, condition)
        emb_negative, masknorm_norm_negative = self.csn(negative, condition)
        emb_positive, masknorm_norm_positive = self.csn(positive, condition)

        mask_norm = (masknorm_norm_anchor + masknorm_norm_negative + masknorm_norm_positive) / 3

        dist_neg = F.pairwise_distance(emb_anchor, emb_negative, 2)
        dist_pos = F.pairwise_distance(emb_anchor, emb_positive, 2)

        return dist_neg, dist_pos, mask_norm