import torch
import torch.nn.functional as F

def compute_bce_loss(inputs, rank_logits, prerank_logits, retrival_logits, device,
                         rank_stage_labels=[1, 0],
                         pre_stage_labels=[1, 0, 0, 0],
                         retr_stage_labels=[1, 0, 0, 0, 0]
                         ):
    mask_list = [tensor.to(device) for tensor in inputs[-10: -5]]
    rank_masks = torch.cat([mask.to(device) for mask in mask_list[:2]], dim=1)
    pre_masks = torch.cat([mask.to(device) for mask in mask_list[:4]], dim=1)
    ret_masks = torch.cat([mask.to(device) for mask in mask_list], dim=1)

    rank_logits = torch.cat([logits.to(device) for logits in rank_logits[:2]], dim=1)
    prerank_logits = torch.cat([logits.to(device) for logits in prerank_logits[:4]], dim=1)
    retrival_logits = torch.cat([logits.to(device) for logits in retrival_logits], dim=1)
    batch_size = retrival_logits.size(0)

    rank_labels_raw = [rank_stage_labels[0]] * 10 + [rank_stage_labels[1]] * 10
    rank_labels = torch.tensor(rank_labels_raw, device=device)
    rank_labels = rank_labels.unsqueeze(0).expand(batch_size, -1)

    prerank_labels_raw = [pre_stage_labels[0]] * 10 + [pre_stage_labels[1]] * 10 + [pre_stage_labels[2]] * 10 + [
        pre_stage_labels[3]] * 10
    prerank_labels = torch.tensor(prerank_labels_raw, device=device)
    prerank_labels = prerank_labels.unsqueeze(0).expand(batch_size, -1)

    retrival_labels_raw = [retr_stage_labels[0]] * 10 + [retr_stage_labels[1]] * 10 + [retr_stage_labels[2]] * 10 + [
        retr_stage_labels[3]] * 10 + [retr_stage_labels[4]] * 10
    retrival_labels = torch.tensor(retrival_labels_raw, device=device)
    retrival_labels = retrival_labels.unsqueeze(0).expand(batch_size, -1)

    rank_loss = F.binary_cross_entropy_with_logits(
        rank_logits,
        rank_labels.float(),
        weight=rank_masks,
        reduction='mean'
    )
    prerank_loss = F.binary_cross_entropy_with_logits(
        prerank_logits,
        prerank_labels.float(),
        weight=pre_masks,
        reduction='mean'
    )
    retrival_loss = F.binary_cross_entropy_with_logits(
        retrival_logits,
        retrival_labels.float(),
        weight=ret_masks,
        reduction='mean'
    )
    return rank_loss, prerank_loss, retrival_loss