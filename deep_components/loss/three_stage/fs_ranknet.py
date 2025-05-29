import torch
import torch.nn.functional as F

def compute_fs_ranknet_loss(masks, logits, device, is_debug=False):
    rerank_pos_logits, rerank_neg_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits = logits

    tmp_logits = torch.cat([rerank_neg_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits], dim=1)
    rerank_pos_bpr_logits = rerank_pos_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)
    rerank_pos_label = masks[0].float().unsqueeze(-1).to(device)  #
    rerank_pos_rank_loss = F.binary_cross_entropy_with_logits(
        rerank_pos_bpr_logits,
        torch.ones(rerank_pos_bpr_logits.size(), dtype=torch.float, device=device),
        weight=torch.ones_like(rerank_pos_label),
        reduction='sum'
    ) / (40 * torch.sum(torch.ones_like(rerank_pos_label)))

    tmp_logits = torch.cat([rank_neg_logits, coarse_neg_logits, prerank_neg_logits], dim=1)
    rerank_neg_bpr_logits = rerank_neg_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)

    rerank_neg_label = masks[1].float().unsqueeze(-1).to(device)  #
    rerank_neg_rank_loss = F.binary_cross_entropy_with_logits(
        rerank_neg_bpr_logits,
        torch.ones(rerank_neg_bpr_logits.size(), dtype=torch.float, device=device),
        weight=torch.ones_like(rerank_neg_label),
        reduction='sum'
    ) / (30 * torch.sum(torch.ones_like(rerank_neg_label)))
    if is_debug: print("rerank_neg_label=%s,%s" % (rerank_neg_label.shape, rerank_neg_label[0]))

    tmp_logits = torch.cat([coarse_neg_logits, prerank_neg_logits], dim=1)
    rank_neg_bpr_logits = rank_neg_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)
    rank_neg_label = masks[2].float().unsqueeze(-1).to(device)  #
    rank_neg_rank_loss = F.binary_cross_entropy_with_logits(
        rank_neg_bpr_logits,
        torch.ones(rank_neg_bpr_logits.size(), dtype=torch.float, device=device),
        weight=torch.ones_like(rank_neg_label),
        reduction='sum'
    ) / (20 * torch.sum(torch.ones_like(rank_neg_label)))
    if is_debug: print("rank_neg_label=%s,%s" % (rank_neg_label.shape, rank_neg_label[0]))

    tmp_logits = prerank_neg_logits
    coarse_neg_bpr_logits = coarse_neg_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)
    coarse_neg_label = masks[3].float().unsqueeze(-1).to(device)  # 21
    if is_debug: print("coarse_neg_label=%s,%s" % (coarse_neg_label.shape, coarse_neg_label[0]))
    coarse_neg_rank_loss = F.binary_cross_entropy_with_logits(
        coarse_neg_bpr_logits,
        torch.ones(coarse_neg_bpr_logits.size(), dtype=torch.float, device=device),
        weight=torch.ones_like(coarse_neg_label),
        reduction='sum'
    ) / (10 * torch.sum(torch.ones_like(coarse_neg_label)))

    loss = rerank_pos_rank_loss + rerank_neg_rank_loss + rank_neg_rank_loss + coarse_neg_rank_loss

    return loss, rerank_pos_rank_loss, rerank_neg_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss