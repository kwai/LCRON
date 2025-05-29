import torch
import torch.nn.functional as F

def compute_bce_metrics(inputs, prerank_logits, retrival_logits, device,
                         pre_stage_labels=[1, 0],
                         retr_stage_labels=[1, 0, 0, 0]
                         ):
    mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]
    pre_masks = torch.cat([mask.to(device) for mask in mask_list[:2]], dim=1)  # shape=[batch_size, total_ad_num]
    ret_masks = torch.cat([mask.to(device) for mask in mask_list], dim=1)  # shape=[batch_size, total_ad_num]
    prerank_logits = torch.cat([logits.to(device) for logits in prerank_logits[:2]],
                               dim=1)  # shape=[batch_size, total_ad_num]
    retrival_logits = torch.cat([logits.to(device) for logits in retrival_logits],
                                dim=1)  # shape=[batch_size, total_ad_num]
    batch_size = retrival_logits.size(0)

    prerank_labels_raw = [pre_stage_labels[0]] * 10 + [pre_stage_labels[1]] * 10
    prerank_labels = torch.tensor(prerank_labels_raw, device=device)
    prerank_labels = prerank_labels.unsqueeze(0).expand(batch_size, -1)

    retrival_labels_raw = [retr_stage_labels[0]] * 10 + [retr_stage_labels[1]] * 10 + [retr_stage_labels[2]] * 10 + [
        retr_stage_labels[3]] * 10  # UPDATE
    print("posr_metric prerank_labels%s,%s" % (prerank_labels.shape, prerank_labels))
    retrival_labels = torch.tensor(retrival_labels_raw, device=device)
    retrival_labels = retrival_labels.unsqueeze(0).expand(batch_size, -1)

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
    return prerank_loss, retrival_loss
