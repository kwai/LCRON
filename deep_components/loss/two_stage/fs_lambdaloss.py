import torch
import torch.nn.functional as F

def compute_lambda_loss(labels_raw, pred_raw, device, smooth_fraction=1.0, topn=20):
    use_rank_idx = True
    if use_rank_idx:
        gain_fn = lambda label: label
    else:
        gain_fn = lambda label: torch.pow(2, label.int()) - 1
    rank_discount_fn = lambda rank: 1 / torch.log(rank + 1)

    pred_raw = pred_raw.to(device)
    # Compute ranks
    ranks_idx = torch.argsort(torch.argsort(pred_raw, dim=1, descending=True), dim=1) + 1  # [pv_cnt, maxquota]

    # Compute gain and normalize with inverse max DCG
    gain = gain_fn(labels_raw).float()
    ideal_sorted_labels = torch.gather(labels_raw, 1, torch.argsort(labels_raw, dim=1, descending=True))
    rank = torch.arange(ideal_sorted_labels.shape[1], device=device) + 1
    discounted_gain = gain_fn(ideal_sorted_labels).float() * rank_discount_fn(rank.float())
    inv_idcg = torch.where(
        discounted_gain.sum(dim=1, keepdim=True) > 0.,
        1. / discounted_gain.sum(dim=1, keepdim=True),
        torch.zeros_like(discounted_gain.sum(dim=1, keepdim=True))
    )
    gain *= inv_idcg

    # Compute pairwise gain
    pair_gain = gain.unsqueeze(2) - gain.unsqueeze(1)  # [pv_cnt, maxquota, maxquota]

    # Compute pair discount
    list_size = labels_raw.shape[1]
    topn = topn or list_size
    rank_diff = torch.abs(ranks_idx.unsqueeze(2) - ranks_idx.unsqueeze(1)).float()
    pair_valid_rank = (ranks_idx.unsqueeze(2) <= topn) | (ranks_idx.unsqueeze(1) <= topn)
    u = torch.where(
        (rank_diff > 0) & pair_valid_rank,
        torch.abs(rank_discount_fn(rank_diff) - rank_discount_fn(rank_diff + 1)),
        torch.zeros_like(rank_diff)
    )

    if smooth_fraction > 0.001:
        rank_discount = torch.where(
            (ranks_idx > topn) | (ranks_idx <= 0),
            torch.zeros_like(ranks_idx.float()),
            rank_discount_fn(ranks_idx.float())
        )
        v = torch.abs(rank_discount.unsqueeze(2) - rank_discount.unsqueeze(1))
        pair_discount = (1. - smooth_fraction) * u + smooth_fraction * v
    else:
        pair_discount = u

    # Compute pair weights
    pair_weight = torch.abs(pair_gain) * pair_discount
    pair_weight = pair_weight.detach()

    # Compute weight loss
    pair_logits = pred_raw.unsqueeze(2) - pred_raw.unsqueeze(1)
    pair_labels = (labels_raw.unsqueeze(2) > labels_raw.unsqueeze(1)).float()
    loss = F.binary_cross_entropy_with_logits(pair_logits, pair_labels, reduction='none')
    weighted_loss = loss * pair_weight
    final_loss = weighted_loss.sum(dim=(-2, -1)).mean(dim=0)
    return final_loss


def compute_fsltr_lambda_all_by_rank(inputs, prerank_logits, retrival_logits, device, is_debug=False, smooth_fraction=1.0, topn=20):
    rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
    all_prerank_logits = torch.cat([logits.to(device) for logits in prerank_logits],
                                   dim=1)  # shape=[batch_size, total_ad_num]
    all_retrival_logits = torch.cat([logits.to(device) for logits in retrival_logits],
                                    dim=1)  # shape=[batch_size, total_ad_num]
    all_rank_index = torch.cat([rank_index.to(device) for rank_index in rank_index_list], dim=1)

    retrival_loss = compute_lambda_loss(all_rank_index, all_retrival_logits, device, smooth_fraction=smooth_fraction, topn=topn)
    prerank_loss = compute_lambda_loss(all_rank_index, all_prerank_logits, device, smooth_fraction=smooth_fraction, topn=topn)

    if is_debug: print(
        "compute_fsltr_lambda_all_by_rank retrival_loss=%s, prerank_loss=%s" % (retrival_loss, prerank_loss))

    loss = retrival_loss + prerank_loss
    outputs = {"total_loss": loss}
    return outputs