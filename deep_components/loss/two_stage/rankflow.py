import torch
import torch.nn.functional as F

def compute_rankflow_metrics(inputs, prerank_logits, retrival_logits, device, alpha=0, is_debug=False):
    rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]

    rank_pos_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index = rank_index_list
    rank_pos_label, rank_neg_label, coarse_label, prerank_label = torch.ones_like(
        rank_pos_rank_index), torch.zeros_like(rank_neg_rank_index), torch.zeros_like(
        coarse_rank_index), torch.zeros_like(prerank_rank_index)

    all_stage_label = torch.cat([rank_pos_label, rank_neg_label, coarse_label, prerank_label], dim=1)
    all_stage_prerank_logits = torch.cat([logits.to(device) for logits in prerank_logits], dim=1)
    all_stage_retrival_logits = torch.cat([logits.to(device) for logits in retrival_logits], dim=1)

    retrieval_loss = F.binary_cross_entropy_with_logits(all_stage_retrival_logits, all_stage_label.float(),
                                                        reduction='mean')
    topk = 30
    _, topk_indices = torch.topk(all_stage_retrival_logits, topk, dim=1)
    if is_debug:
        print(f"DEBUG: all_stage_retrival_logits:\n{all_stage_retrival_logits.detach().cpu().numpy()}")
        print(f"DEBUG: topk_indices:\n{topk_indices.detach().cpu().numpy()}")

    topk_retrieval_labels = torch.gather(all_stage_label, 1, topk_indices)
    topk_prerank_logits = torch.gather(all_stage_prerank_logits, 1, topk_indices)
    topk_retrieval_logits = torch.gather(all_stage_retrival_logits, 1, topk_indices)

    prerank_loss = F.binary_cross_entropy_with_logits(topk_prerank_logits, topk_retrieval_labels.float(),
                                                      reduction='mean')
    mse_loss = F.mse_loss(topk_prerank_logits.detach(), topk_retrieval_logits)
    print("distillation_loss=%s,%s" % (mse_loss.shape, mse_loss))

    prerank_pos_topk = 10
    _, prerank_pos_topk_indices = torch.topk(topk_prerank_logits, prerank_pos_topk, dim=1)
    prerank_neg_topk = 20
    _, prerank_neg_topk_indices = torch.topk(-topk_prerank_logits, prerank_neg_topk, dim=1)

    retrieval_pos_topk_logits = torch.gather(topk_retrieval_logits, 1, prerank_pos_topk_indices)
    retrieval_neg_topk_logits = torch.gather(topk_retrieval_logits, 1, prerank_neg_topk_indices)
    retrieval_pos_topk_mean_logits = torch.mean(retrieval_pos_topk_logits, dim=-1)
    retrieval_neg_topk_mean_logits = torch.mean(retrieval_neg_topk_logits, dim=-1)

    rank_loss = -torch.mean(F.logsigmoid(retrieval_pos_topk_mean_logits - retrieval_neg_topk_mean_logits))
    print("rank_loss=%s,%s" % (rank_loss.shape, rank_loss))

    total_loss = retrieval_loss + prerank_loss + (1 - alpha) * mse_loss + alpha * rank_loss
    if is_debug:
        print("DEBUG_RankFlow_LOSS. %s\t%s\t%s\t" % (retrieval_loss.detach().cpu().numpy(),
                                                     prerank_loss.detach().cpu().numpy(),
                                                     mse_loss.detach().cpu().numpy()
                                                     ))

    outputs = {"total_loss": total_loss}
    return outputs
