import torch
import torch.nn.functional as F

joint_loss_conf = type("", (), {
    "rank_model_name": 'joint/rank_model',
    "prerank_model_name": 'joint/prerank_model',
    "recall_model_name": 'joint/recall_model',
    "joint_recall_k": 40,
    "joint_prerank_k": 30,
    "joint_rank_k": 20,
    "k_ls": "",
    "gt_num": 10,
    "global_size": 50})

def tensor_concat(logits_list, rank_index_list, mask_list, device):
    mask_tensor = torch.cat([mask.to(device) for mask in mask_list], dim=1)  # shape=[batch_size, total_ad_num]
    logits_tensor = torch.cat([logits.to(device) for logits in logits_list], dim=1)  # shape=[batch_size, total_ad_num]
    rank_index_tensor = torch.cat([rank_index.to(device) for rank_index in rank_index_list],
                                  dim=1)  # shape=[batch_size, total_ad_num]
    mask_sum_per_pv = mask_tensor.sum(dim=1)

    return logits_tensor, rank_index_tensor, mask_tensor, None, mask_sum_per_pv

def compute_icc_loss(inputs, rank_logits, prerank_logits, retrival_logits, device):
    delta, sigma = 1e-3, 1e-3
    rank_index_list = [tensor.to(device) for tensor in inputs[-5:]]
    mask_list = [tensor.to(device) for tensor in inputs[-10:-5]]

    rerank_pos_rank_index, rerank_neg_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index = rank_index_list  # shape=[batch_size, ad_num_per_pv]
    rerank_pos_mask, rerank_neg_mask, rank_neg_mask, coarse_mask, prerank_mask = mask_list  # shape=[batch_size, ad_num_per_pv]

    rank_sorted_logits, sorted_rank_index_list, _, _, mask_sum_per_pv_list = tensor_concat(
        logits_list=rank_logits,
        rank_index_list=[rerank_pos_rank_index, rerank_neg_rank_index, rank_neg_rank_index, coarse_rank_index,
                         prerank_rank_index],
        mask_list=[rerank_pos_mask, rerank_neg_mask, rank_neg_mask, coarse_mask, prerank_mask],
        device=device)

    prerank_sorted_logits, _, _, _, _ = tensor_concat(
        logits_list=prerank_logits,
        rank_index_list=[rerank_pos_rank_index, rerank_neg_rank_index, rank_neg_rank_index, coarse_rank_index,
                         prerank_rank_index],
        mask_list=[rerank_pos_mask, rerank_neg_mask, rank_neg_mask, coarse_mask, prerank_mask],
        device=device)

    retrival_sorted_logits, _, _, _, _ = tensor_concat(
        logits_list=retrival_logits,
        rank_index_list=[rerank_pos_rank_index, rerank_neg_rank_index, rank_neg_rank_index, coarse_rank_index,
                         prerank_rank_index],
        mask_list=[rerank_pos_mask, rerank_neg_mask, rank_neg_mask, coarse_mask, prerank_mask],
        device=device)

    count = mask_sum_per_pv_list
    count = count.to(device)

    model_outputs_dict = {
        "joint/rank_model": {
            "grouped_logits": rank_sorted_logits
        },
        "joint/prerank_model": {
            "grouped_logits": prerank_sorted_logits
        },
        "joint/recall_model": {
            "grouped_logits": retrival_sorted_logits
        }
    }

    def kappa(stage_score, cutoff_point):
        _, indices = torch.topk(stage_score, cutoff_point, dim=-1, largest=True, sorted=True)
        kth_largest = torch.gather(stage_score, 1, indices[:, -1].unsqueeze(1))
        return kth_largest.expand_as(stage_score)

    def indicator_func(predict, kappa_value, indicator_type="Logistic", delta=1e-3, sigma=1e-3):
        if indicator_type == "Logistic":
            return torch.sigmoid((predict - kappa_value) / sigma)
        elif indicator_type == "Relu":
            return 0.5 * (1 + torch.clamp((predict - kappa_value) / delta, min=-1.0, max=1.0))
        else:
            return torch.ones_like(predict)

    grouped_ensemble_logits = torch.zeros_like(model_outputs_dict['joint/rank_model']['grouped_logits'])
    cum_indicator_score = torch.ones_like(model_outputs_dict['joint/rank_model']['grouped_logits'])

    recall_logits = model_outputs_dict['joint/recall_model']['grouped_logits']
    kappa_value = kappa(recall_logits, joint_loss_conf.joint_recall_k)
    indicator_score = indicator_func(recall_logits, kappa_value, delta=delta, sigma=sigma)
    grouped_ensemble_logits = grouped_ensemble_logits + (1 - indicator_score) * recall_logits
    cum_indicator_score = cum_indicator_score * indicator_score

    prerank_logits = model_outputs_dict['joint/prerank_model']['grouped_logits']
    kappa_value = kappa(prerank_logits, joint_loss_conf.joint_prerank_k)
    indicator_score = indicator_func(prerank_logits, kappa_value, delta=delta, sigma=sigma)
    grouped_ensemble_logits = grouped_ensemble_logits + cum_indicator_score * 2 * (1 - indicator_score) * prerank_logits
    cum_indicator_score = cum_indicator_score * indicator_score

    rank_logits = model_outputs_dict['joint/rank_model']['grouped_logits']
    kappa_value = kappa(rank_logits, joint_loss_conf.joint_prerank_k)
    indicator_score = indicator_func(rank_logits, kappa_value, delta=delta, sigma=sigma)
    grouped_ensemble_logits = grouped_ensemble_logits + cum_indicator_score * 4 * indicator_score * rank_logits

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