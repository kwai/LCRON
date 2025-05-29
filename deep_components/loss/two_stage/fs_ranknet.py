import torch
import torch.nn.functional as F


def compute_fsltr_loss_all_by_rank(masks, logits, args, device, is_debug=False):
    rank_pos_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits = logits

    tmp_logits = torch.cat([rank_neg_logits, coarse_neg_logits, prerank_neg_logits], dim=1)
    rank_pos_bpr_logits = rank_pos_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)  # b*10*30
    rank_pos_label = masks[0].float().unsqueeze(-1).to(device) #
    rank_pos_rank_loss = F.binary_cross_entropy_with_logits(
        rank_pos_bpr_logits,
        torch.ones(rank_pos_bpr_logits.size(), dtype=torch.float, device=device),
        weight=rank_pos_label,
        reduction='sum'
    ) / (30 * torch.sum(rank_pos_label))
    if is_debug: print("rank_pos_bpr_logits=%s,%s" %(rank_pos_bpr_logits.shape, rank_pos_bpr_logits[0]))
    if is_debug: print("rank_pos_label=%s,%s" %(rank_pos_label.shape, rank_pos_label[0]))

    tmp_logits = torch.cat([coarse_neg_logits, prerank_neg_logits], dim=1)
    rank_neg_bpr_logits = rank_neg_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)  # b*10*20
    rank_neg_label = masks[1].float().unsqueeze(-1).to(device)  #
    rank_neg_rank_loss = F.binary_cross_entropy_with_logits(
        rank_neg_bpr_logits,
        torch.ones(rank_neg_bpr_logits.size(), dtype=torch.float, device=device),
        weight=rank_neg_label,
        reduction='sum'
    ) / (20 * torch.sum(rank_neg_label))
    if is_debug: print("rank_neg_label=%s,%s" %(rank_neg_label.shape, rank_neg_label[0]))

    tmp_logits = prerank_neg_logits
    coarse_neg_bpr_logits = coarse_neg_logits.unsqueeze(-1) - tmp_logits.unsqueeze(1)  # b*10*10
    coarse_neg_label = masks[2].float().unsqueeze(-1).to(device) # 21
    if is_debug: print("coarse_neg_label=%s,%s" %(coarse_neg_label.shape, coarse_neg_label[0]))
    coarse_neg_rank_loss = F.binary_cross_entropy_with_logits(
        coarse_neg_bpr_logits,
        torch.ones(coarse_neg_bpr_logits.size(), dtype=torch.float, device=device),
        weight=coarse_neg_label,
        reduction='sum'
    ) / (10 * torch.sum(coarse_neg_label))

    loss = rank_pos_rank_loss + rank_neg_rank_loss + coarse_neg_rank_loss
    return loss, rank_pos_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss
