import torch
import torch.nn.functional as F
from deep_components.loss.loss_helper_base import LossHelperBase
from deep_components.loss.sorting import neuralsort


def bl_matmul(A, B):
    return torch.einsum('mij,jk->mik', A, B)


def get_shape_at_axis(tensor, axis):
    if tensor is None:
        return 0
    try:
        return tensor.shape[axis]
    except IndexError:
        return 0


def sequence_mask(count, padding_to_len):
    range_tensor = torch.arange(padding_to_len, device=count.device).unsqueeze(0)
    range_tensor = range_tensor.expand(count.size(0), padding_to_len)
    mask = (count.unsqueeze(-1) > range_tensor).float()
    return mask


def tensor_concat(logits_list, rank_index_list, mask_list, device):
    mask_tensor = torch.cat([mask.to(device) for mask in mask_list], dim=1)  # shape=[batch_size, total_ad_num]
    logits_tensor = torch.cat([logits.to(device) for logits in logits_list], dim=1)  # shape=[batch_size, total_ad_num]
    rank_index_tensor = torch.cat([rank_index.to(device) for rank_index in rank_index_list],
                                  dim=1)  # shape=[batch_size, total_ad_num]
    mask_sum_per_pv = mask_tensor.sum(dim=1)
    return logits_tensor, rank_index_tensor, mask_tensor, None, mask_sum_per_pv


def get_set_value_by_permutation_matrix_and_label(permutation_matrix, label, top_k):
    t = torch.matmul(permutation_matrix, label.unsqueeze(-1)).squeeze(-1)  # [batch_size, N]
    value = torch.sum(t[:, :top_k], dim=-1)  # [batch_size]
    optimal_value, _ = torch.topk(label, k=top_k, dim=-1)
    set_value_sample_wise = value / torch.sum(optimal_value, dim=-1)
    return torch.mean(set_value_sample_wise)


def compute_arf_metrics(inputs, prerank_logits, retrival_logits, device, max_num, prerank_loss_conf, retrival_loss_conf,
                        logger,
                        loss_model_prerank=None,
                        loss_model_retrival=None,
                        loss_type="arf"
                        ):
    tau = 200
    rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
    mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]

    rank_pos_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index = rank_index_list  # shape=[batch_size, ad_num_per_pv]
    rank_pos_mask, rank_neg_mask, coarse_mask, prerank_mask = mask_list  # shape=[batch_size, ad_num_per_pv]

    prerank_sorted_logits, sorted_rank_index_list, _, _, mask_sum_per_pv_list = tensor_concat(
        logits_list=prerank_logits,
        rank_index_list=[rank_pos_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index],
        mask_list=[rank_pos_mask, rank_neg_mask, coarse_mask, prerank_mask],
        device=device)

    retrival_sorted_logits, _, _, _, _ = tensor_concat(
        logits_list=retrival_logits,
        rank_index_list=[rank_pos_rank_index, rank_neg_rank_index, coarse_rank_index, prerank_rank_index],
        mask_list=[rank_pos_mask, rank_neg_mask, coarse_mask, prerank_mask],
        device=device)

    count = mask_sum_per_pv_list
    count = count.to(device)
    label_mask = sequence_mask(count, max_num)

    label_permutation_matrix = neuralsort(sorted_rank_index_list.float(), 0.0001)
    grouped_labels = label_permutation_matrix
    label_infos = {
        "label_permutation_matrix": label_permutation_matrix,
        "label_mask": label_mask,
        "grouped_labels": grouped_labels,
        "count": count
    }

    prerank_logits_matrix = neuralsort(prerank_sorted_logits, tau)
    model_outputs_dict = {
        "prerank_model": {
            "logits_permutation_matrix": prerank_logits_matrix
        }
    }

    if loss_type == "arf_v2":
        loss_prerank = ARFV2Loss(name='prerank_model',
                               label_infos=label_infos,
                               model_outputs=model_outputs_dict,
                               loss_conf=prerank_loss_conf,
                               loss_model=loss_model_prerank,
                               logger=logger,
                               use_name_as_scope=True,
                               device=device,
                               is_debug=True,
                               is_train=True)
    else:
        loss_prerank = ARFLoss(name='prerank_model',
                               label_infos=label_infos,
                               model_outputs=model_outputs_dict,
                               loss_conf=prerank_loss_conf,
                               loss_model=loss_model_prerank,
                               logger=logger,
                               use_name_as_scope=True,
                               device=device,
                               is_debug=True,
                               is_train=True)

    prerank_loss = loss_prerank.get_loss('total_loss')

    retrival_matrix = neuralsort(retrival_sorted_logits, tau)
    model_outputs_dict = {
        "retrival_model": {
            "logits_permutation_matrix": retrival_matrix
        }
    }

    if loss_type == "arf_v2":
        loss_retrival = ARFV2Loss(name='retrival_model',
                                label_infos=label_infos,
                                model_outputs=model_outputs_dict,
                                loss_conf=retrival_loss_conf,
                                loss_model=loss_model_retrival,
                                logger=logger,
                                use_name_as_scope=True,
                                device=device,
                                is_debug=True,
                                is_train=True)
    else:  # ARF
        loss_retrival = ARFLoss(name='retrival_model',
                                label_infos=label_infos,
                                model_outputs=model_outputs_dict,
                                loss_conf=retrival_loss_conf,
                                loss_model=loss_model_retrival,
                                logger=logger,
                                use_name_as_scope=True,
                                device=device,
                                is_debug=True,
                                is_train=True)
    retrival_loss = loss_retrival.get_loss('total_loss')
    outputs = {
        "total_loss": prerank_loss + retrival_loss,
        "prerank_loss": prerank_loss,
        "retrival_loss": retrival_loss,
    }
    return outputs


class LRelaxLoss(LossHelperBase):
    def loss_graph(self):
        top_k = self.conf.top_k
        support_m = self.conf.support_m
        permutation_matrix = self.model_outputs[self.conf.model_name]['logits_permutation_matrix']
        label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = mask_all.unsqueeze(1) * mask_all.unsqueeze(2)
        permutation_matrix = permutation_matrix * s2_mask
        label_matrix = label_matrix * s2_mask
        target_permutation_matrix = torch.mean(permutation_matrix[:, :support_m, :], dim=-2)
        target_label_matrix = torch.sum(label_matrix[:, :top_k, :], dim=-2)
        loss_sample_wise = torch.mean(
            -torch.log(target_permutation_matrix + 1e-6) * target_label_matrix * self.label_infos['label_mask'], dim=-1)
        loss_sample_wise = loss_sample_wise * (self.label_infos['count'] > support_m).float()
        if hasattr(self.conf, 'sample_weight'):
            loss = torch.mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = torch.mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            acc = get_set_value_by_permutation_matrix_and_label(
                self.model_outputs[self.conf.model_name]['logits_permutation_matrix'],
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['acc'] = acc

    def _init_check(self):
        assert hasattr(self.conf, 'top_k'), '{ERROR} %s missing argument joint_recall_k' % self.name
        assert hasattr(self.conf, 'support_m'), '{ERROR} %s missing argument support_m' % self.name
        assert hasattr(self.conf, 'model_name'), '{ERROR} %s missing argument model_name' % self.name
        assert 'label_permutation_matrix' in self.label_infos, '{ERROR} %s missing label: label_permutation_matrix' % self.name


class LRelaxUltraLoss(LossHelperBase):
    def loss_graph(self):
        top_k = self.conf.top_k
        support_m = self.conf.support_m
        permutation_matrix = self.model_outputs[self.conf.model_name]['logits_permutation_matrix']
        label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = mask_all.unsqueeze(1) * mask_all.unsqueeze(2)
        permutation_matrix = permutation_matrix * s2_mask
        label_matrix = label_matrix * s2_mask
        detach_permutation_matrix = permutation_matrix.detach()
        up_target_permutation_matrix = torch.sum(permutation_matrix[:, :support_m, :], dim=-2)
        raw_sum_permutation_matrix = torch.sum(detach_permutation_matrix, dim=-2)
        up_target_permutation_matrix = up_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
        up_target_label_matrix = torch.sum(label_matrix[:, :top_k, :], dim=-2)
        up_loss_sample_wise = torch.mean(
            -torch.log(up_target_permutation_matrix + 1e-6) * up_target_label_matrix * self.label_infos['label_mask'],
            dim=-1)
        down_target_permutation_matrix = torch.sum(permutation_matrix[:, support_m:, :], dim=-2)
        down_target_permutation_matrix = down_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
        down_target_label_matrix = torch.sum(label_matrix[:, top_k:, :], dim=-2)
        down_loss_sample_wise = torch.mean(
            -torch.log(down_target_permutation_matrix + 1e-6) * down_target_label_matrix * self.label_infos[
                'label_mask'], dim=-1)
        loss_sample_wise = up_loss_sample_wise + down_loss_sample_wise
        loss_sample_wise = loss_sample_wise * (self.label_infos['count'] > support_m).float()
        if hasattr(self.conf, 'sample_weight'):
            loss = torch.mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = torch.mean(loss_sample_wise)

        self.loss_output_dict[self.name] = loss


class GlobalLoss(LossHelperBase):
    def loss_graph(self):
        permutation_matrix = self.model_outputs[self.conf.model_name]['logits_permutation_matrix']
        label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = mask_all.unsqueeze(1) * mask_all.unsqueeze(2)
        permutation_matrix = permutation_matrix * s2_mask
        label_matrix = label_matrix * s2_mask
        top_k = self.conf.global_size
        loss_sample_wise = torch.mean(
            torch.mean(-torch.log(permutation_matrix[:, :top_k, :] + 1e-6) * label_matrix[:, :top_k, :], dim=-1),
            dim=-1)
        loss_sample_wise = loss_sample_wise * (self.label_infos['count'] >= top_k).float()
        if hasattr(self.conf, 'sample_weight'):
            loss = torch.mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = torch.mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss


class ARFLoss(LossHelperBase):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False,
                 is_train=True, device=None, loss_model=None):
        super(ARFLoss, self).__init__(name=name, label_infos=label_infos, model_outputs=model_outputs,
                                      loss_conf=loss_conf,
                                      logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                      is_train=is_train)
        self.l_relax_helper = LRelaxLoss(name=self.name + '/L_relax', label_infos=label_infos,
                                         model_outputs=model_outputs, loss_conf=loss_conf,
                                         logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                         is_train=is_train)
        self.global_loss_helper = GlobalLoss(name=self.name + '/L_global', label_infos=label_infos,
                                             model_outputs=model_outputs, loss_conf=loss_conf,
                                             logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                             is_train=is_train)
        self.device = device
        self.loss_conf = loss_conf
        self.loss_model = loss_model

    def loss_graph(self):
        l_relax = self.l_relax_helper.get_loss(self.l_relax_helper.name)
        l_global = self.global_loss_helper.get_loss(self.global_loss_helper.name)
        final_loss = self.loss_model.forward(l_global, l_relax)
        self.global_weight = self.loss_model.global_weight
        self.local_weight = self.loss_model.local_weight

        self.loss_output_dict['global_loss'] = l_global
        self.loss_output_dict['global_weight'] = self.global_weight
        self.loss_output_dict['local_loss'] = l_relax
        self.loss_output_dict['local_weight'] = self.local_weight
        self.loss_output_dict[self.name] = final_loss.squeeze()
        self.loss_output_dict['total_loss'] = final_loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            acc = get_set_value_by_permutation_matrix_and_label(
                self.model_outputs[self.conf.model_name]['logits_permutation_matrix'],
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['acc'] = acc


class ARFV2Loss(LossHelperBase):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False,
                 is_train=True, device=None, loss_model=None):
        super(ARFV2Loss, self).__init__(name=name, label_infos=label_infos, model_outputs=model_outputs,
                                        loss_conf=loss_conf,
                                        logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                        is_train=is_train)
        self.l_relax_helper = LRelaxUltraLoss(name=self.name + '/L_relax', label_infos=label_infos,
                                              model_outputs=model_outputs, loss_conf=loss_conf,
                                              logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                              is_train=is_train)
        self.global_loss_helper = GlobalLoss(name=self.name + '/L_global', label_infos=label_infos,
                                             model_outputs=model_outputs, loss_conf=loss_conf,
                                             logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                             is_train=is_train)
        self.device = device
        self.loss_conf = loss_conf
        self.loss_model = loss_model

    def loss_graph(self):
        l_relax = self.l_relax_helper.get_loss(self.l_relax_helper.name)
        l_global = self.global_loss_helper.get_loss(self.global_loss_helper.name)
        final_loss = self.loss_model.forward(l_global, l_relax)
        self.global_weight = self.loss_model.global_weight
        self.local_weight = self.loss_model.local_weight

        self.loss_output_dict['global_loss'] = l_global
        self.loss_output_dict['global_weight'] = self.global_weight
        self.loss_output_dict['local_loss'] = l_relax
        self.loss_output_dict['local_weight'] = self.local_weight
        self.loss_output_dict[self.name] = final_loss.squeeze()
        self.loss_output_dict['total_loss'] = final_loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            acc = get_set_value_by_permutation_matrix_and_label(
                self.model_outputs[self.conf.model_name]['logits_permutation_matrix'],
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['acc'] = acc


class ArfLossModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.global_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=self.device))
        self.local_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=self.device))

    def forward(self, l_global, l_relax):
        final_loss = (0.5 / torch.square(self.global_weight)) * l_global + l_relax + torch.log(self.global_weight)
        print("DEBUG_ArfLossModel. global_weight=%s l_global=%s l_relax=%s" % (self.global_weight, l_global, l_relax))
        return final_loss