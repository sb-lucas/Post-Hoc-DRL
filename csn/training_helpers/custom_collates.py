from torch.nn.utils.rnn import pad_sequence
import torch

def attention_train_collate_fn(batch):
    embs, i_vecs, g_vecs, m_vecs, artist_ids = zip(*batch)

    lengths = torch.tensor([e.shape[0] for e in embs])
    padded = pad_sequence(embs, batch_first=True)          # (B, N_max, 768)
    N_max = padded.shape[1]
    attn_mask = torch.arange(N_max)[None, :] < lengths[:, None]  # True = real frame

    i_vecs = torch.stack(i_vecs)
    g_vecs = torch.stack(g_vecs)
    m_vecs = torch.stack(m_vecs)
    artist_ids = torch.stack(artist_ids)

    return (padded, attn_mask), i_vecs, g_vecs, m_vecs, artist_ids

def attention_val_collate_fn(batch):
    emb_anchors, emb_negatives, emb_positives, conditions = zip(*batch)

    def pad_and_mask(embs):
        lengths = torch.tensor([e.shape[0] for e in embs])
        padded = pad_sequence(embs, batch_first=True)
        N_max = padded.shape[1]
        mask = torch.arange(N_max)[None, :] < lengths[:, None]
        return padded, mask

    anchor_padded, anchor_mask = pad_and_mask(emb_anchors)
    negative_padded, negative_mask = pad_and_mask(emb_negatives)
    positive_padded, positive_mask = pad_and_mask(emb_positives)
    conditions = torch.tensor(conditions)

    return ((anchor_padded, anchor_mask), (negative_padded, negative_mask),
            (positive_padded, positive_mask), conditions)

def mean_train_collate_fn(batch):
    embs, i_vecs, g_vecs, m_vecs, artist_ids = zip(*batch)
    embs = torch.stack(embs)  # (B, 768)
    i_vecs = torch.stack(i_vecs)
    g_vecs = torch.stack(g_vecs)
    m_vecs = torch.stack(m_vecs)
    artist_ids = torch.stack(artist_ids)
    return (embs, None), i_vecs, g_vecs, m_vecs, artist_ids

def mean_val_collate_fn(batch):
    emb_anchors, emb_negatives, emb_positives, conditions = zip(*batch)
    emb_anchors = torch.stack(emb_anchors)
    emb_negatives = torch.stack(emb_negatives)
    emb_positives = torch.stack(emb_positives)
    conditions = torch.tensor(conditions)
    return ((emb_anchors, None), (emb_negatives, None), (emb_positives, None), conditions)