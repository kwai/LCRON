import torch
import torch.nn.functional as F

def compute_rankflow_loss(inputs, rank_logits, prerank_logits, retrieval_logits, device):
    rank_index_list = [tensor.to(device) for tensor in inputs[-5:]]
    rerank_pos_rank_index, rerank_neg_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index = rank_index_list

    rerank_pos_label, rerank_neg_label, rank_neg_label, coarse_label, prerank_label = torch.ones_like(
        rerank_pos_rank_index), torch.zeros_like(rerank_neg_rank_index), torch.zeros_like(
        rank_neg_rank_index), torch.zeros_like(coarse_rank_index), torch.zeros_like(prerank_rank_index)

    all_stage_label = torch.cat([rerank_pos_label, rerank_neg_label, rank_neg_label, coarse_label, prerank_label],
                                dim=1)
    all_stage_rank_logits = torch.cat([logits.to(device) for logits in rank_logits], dim=1)
    all_stage_prerank_logits = torch.cat([logits.to(device) for logits in prerank_logits], dim=1)
    all_stage_retrieval_logits = torch.cat([logits.to(device) for logits in retrieval_logits], dim=1)

    retrieval_loss = F.binary_cross_entropy_with_logits(all_stage_retrieval_logits, all_stage_label.float(),
                                                        reduction='mean')

    topk = 40
    _, topk_indices = torch.topk(all_stage_retrieval_logits, topk, dim=1)

    topk_retrieval_labels = torch.gather(all_stage_label, 1, topk_indices)
    topk_rank_logits = torch.gather(all_stage_rank_logits, 1, topk_indices)
    topk_prerank_logits = torch.gather(all_stage_prerank_logits, 1, topk_indices)
    topk_retrieval_logits = torch.gather(all_stage_retrieval_logits, 1, topk_indices)
    prerank_loss = F.binary_cross_entropy_with_logits(topk_prerank_logits, topk_retrieval_labels.float(),
                                                      reduction='mean')
    retrieval_mse_loss = F.mse_loss(topk_prerank_logits.detach(), topk_retrieval_logits)

    prerank_pos_topk = 30
    _, prerank_pos_topk_indices = torch.topk(topk_prerank_logits, prerank_pos_topk, dim=1)
    prerank_neg_topk = 10
    _, prerank_neg_topk_indices = torch.topk(-topk_prerank_logits, prerank_neg_topk, dim=1)
    retrieval_pos_topk_logits = torch.gather(topk_retrieval_logits, 1, prerank_pos_topk_indices)
    retrieval_neg_topk_logits = torch.gather(topk_retrieval_logits, 1, prerank_neg_topk_indices)
    retrieval_pos_topk_mean_logits = torch.mean(retrieval_pos_topk_logits, dim=-1)
    retrieval_neg_topk_mean_logits = torch.mean(retrieval_neg_topk_logits, dim=-1)
    retrieval_rank_loss = -torch.mean(F.logsigmoid(retrieval_pos_topk_mean_logits - retrieval_neg_topk_mean_logits))

    # 30
    prerank_topk_rank_logits = torch.gather(topk_rank_logits, 1, prerank_pos_topk_indices)
    prerank_topk_prerank_logits = torch.gather(topk_prerank_logits, 1, prerank_pos_topk_indices)
    topk_prerank_labels = torch.gather(topk_retrieval_labels, 1, prerank_pos_topk_indices)
    rank_loss = F.binary_cross_entropy_with_logits(prerank_topk_rank_logits, topk_prerank_labels.float(),
                                                   reduction='mean')
    prerank_mse_loss = F.mse_loss(prerank_topk_rank_logits.detach(), prerank_topk_prerank_logits)

    rank_pos_topk = 20
    _, rank_pos_topk_indices = torch.topk(prerank_topk_rank_logits, rank_pos_topk, dim=1)
    rank_neg_topk = 10
    _, rank_neg_topk_indices = torch.topk(-prerank_topk_rank_logits, rank_neg_topk, dim=1)
    prerank_pos_topk_logits = torch.gather(prerank_topk_prerank_logits, 1, rank_pos_topk_indices)
    prerank_neg_topk_logits = torch.gather(prerank_topk_prerank_logits, 1, rank_neg_topk_indices)
    prerank_pos_topk_mean_logits = torch.mean(prerank_pos_topk_logits, dim=-1)
    prerank_neg_topk_mean_logits = torch.mean(prerank_neg_topk_logits, dim=-1)
    prerank_rank_loss = -torch.mean(F.logsigmoid(prerank_pos_topk_mean_logits - prerank_neg_topk_mean_logits))

    total_loss = retrieval_loss + prerank_loss + rank_loss + retrieval_mse_loss + prerank_mse_loss + retrieval_rank_loss + prerank_rank_loss

    outputs = {"total_loss": total_loss}
    return outputs