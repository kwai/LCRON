import torch
import torch.nn.functional as F

def tensor_concat(logits_list, rank_index_list, mask_list, device):
    mask_tensor = torch.cat([mask.to(device) for mask in mask_list], dim=1)  # shape=[batch_size, total_ad_num]
    logits_tensor = torch.cat([logits.to(device) for logits in logits_list], dim=1)  # shape=[batch_size, total_ad_num]
    rank_index_tensor = torch.cat([rank_index.to(device) for rank_index in rank_index_list],
                                  dim=1)  # shape=[batch_size, total_ad_num]
    mask_sum_per_pv = mask_tensor.sum(dim=1)
    return logits_tensor, rank_index_tensor, mask_tensor, None, mask_sum_per_pv

def compute_icc_metrics(inputs, prerank_logits, retrival_logits, device, joint_loss_conf):
    sigma, delta = 1e-3, 1e-3
    rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
    mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]

    prerank_sorted_logits, sorted_rank_index_list, _, _, mask_sum_per_pv_list = tensor_concat(
        logits_list=prerank_logits,
        rank_index_list=rank_index_list,
        mask_list=mask_list,
        device=device)

    retrival_sorted_logits, _, _, _, _ = tensor_concat(
        logits_list=retrival_logits,
        rank_index_list=rank_index_list,
        mask_list=mask_list,
        device=device)

    count = mask_sum_per_pv_list
    count = count.to(device)

    model_outputs_dict = {
        "joint/prerank_model": {
            "grouped_logits": prerank_sorted_logits
        },
        "joint/recall_model": {
            "grouped_logits": retrival_sorted_logits
        }
    }

    def kappa(stage_score, cutoff_point):
        _, indices = torch.topk(stage_score, cutoff_point, dim=-1)
        kth_largest = torch.gather(stage_score, 1, indices.select(1, -1).unsqueeze(1))
        kth_largest = kth_largest.expand_as(stage_score)
        return kth_largest

    def indicator_func(predict, kappa_value, indicator_type="Logistic", delta=1e-3, sigma=1e-3):
        if indicator_type == "Logistic":
            return torch.sigmoid((predict - kappa_value) / sigma)
        elif indicator_type == "Relu":
            return 0.5 * (1 + torch.clamp((predict - kappa_value) / delta, min=-1.0, max=1.0))
        else:
            return torch.ones_like(predict)

    grouped_ensemble_logits = torch.zeros_like(model_outputs_dict['joint/prerank_model']['grouped_logits'])
    cum_indicator_score = torch.ones_like(model_outputs_dict['joint/prerank_model']['grouped_logits'])

    kappa_value = kappa(model_outputs_dict['joint/recall_model']['grouped_logits'], joint_loss_conf.joint_recall_k)
    indicator_score = indicator_func(model_outputs_dict['joint/recall_model']['grouped_logits'], kappa_value,
                                     delta=delta, sigma=sigma)
    grouped_ensemble_logits += (1 - indicator_score) * model_outputs_dict['joint/recall_model']['grouped_logits']
    cum_indicator_score *= indicator_score

    kappa_value = kappa(model_outputs_dict['joint/prerank_model']['grouped_logits'], joint_loss_conf.joint_recall_k)
    indicator_score = indicator_func(model_outputs_dict['joint/prerank_model']['grouped_logits'], kappa_value,
                                     delta=delta, sigma=sigma)
    grouped_ensemble_logits += cum_indicator_score * indicator_score * model_outputs_dict['joint/prerank_model'][
        'grouped_logits']

    grouped_labels = sorted_rank_index_list.float()
    n = grouped_labels.size(1)
    upper_triangular_matrix = torch.triu(torch.ones(n, n, device=grouped_labels.device), diagonal=1)
    group_masks = upper_triangular_matrix

    pair_logits = grouped_ensemble_logits.unsqueeze(-1) - grouped_ensemble_logits.unsqueeze(1)
    pair_labels = (grouped_labels.unsqueeze(-1) > grouped_labels.unsqueeze(1)).float()

    loss = F.binary_cross_entropy_with_logits(pair_logits, pair_labels, reduction='none')
    loss = (loss * group_masks).mean()
    outputs = {"total_loss": loss}

    return outputs
