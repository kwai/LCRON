import itertools
import sys
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from models import DSSM, DIN
from dataset2 import Rank_Train_All_BY_RANK_Dataset
from utils import load_pkl

from deep_components.loss.two_stage.icc import compute_icc_metrics
from deep_components.loss.two_stage.bce import compute_bce_metrics
from deep_components.loss.two_stage.arf import compute_arf_metrics, ArfLossModel
from deep_components.loss.two_stage.fs_ranknet import compute_fsltr_loss_all_by_rank
from deep_components.loss.two_stage.fs_lambdaloss import compute_fsltr_lambda_all_by_rank
from deep_components.loss.two_stage.rankflow import compute_rankflow_metrics
from deep_components.loss.two_stage.lcron import compute_lcron_metrics, LcronLossModel

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

### GLOBAL
# rank_pos(10) + rank_neg(10) + coarse_neg(10) + prerank_neg(10)
max_num = 40
joint_loss_conf = type("", (), {"prerank_model_name": 'joint/prerank_model',
                                "recall_model_name": 'joint/recall_model',
                                "joint_recall_k": 30,
                                "joint_prerank_k": 20,
                                "gt_num": 10,
                                "global_size": max_num})

# conf for ARF:
prerank_arf_loss_conf = type("", (), {
    "model_name": 'prerank_model',
    "top_k": 10,
    "support_m": 20,
    "gt_num": 10,
    "global_size": max_num,
})

retrival_arf_loss_conf = type("", (), {
    "model_name": 'retrival_model',
    "top_k": 10,
    "support_m": 30,
    "gt_num": 10,
    "global_size": max_num,
})


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1, help='epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='train batch size.')
    parser.add_argument('--infer_realshow_batch_size', type=int, default=1024, help='inference batch size.')
    parser.add_argument('--infer_recall_batch_size', type=int, default=1024, help='inference batch size.')
    parser.add_argument('--nn_units', type=int, default=128, help='nn units.')
    parser.add_argument('--emb_dim', type=int, default=8, help='embedding dimension.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--seq_len', type=int, default=3, help='length of behaivor sequence')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device.')
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of print.')
    parser.add_argument('--tag', type=str, default="1st", help='exp tag.')
    parser.add_argument('--root_path', type=str, default=".", help='root path to data, checkpoints and logs')
    parser.add_argument('--tau', type=float, default=1, help='tau.')
    parser.add_argument('--loss_type', type=str, default="fsltr", help='method type.')

    return parser.parse_args()


def print_model(model, desc, num_limit=10):
    kv_ls = [(name, param) for name, param in model.named_parameters()]
    kv_ls.sort(key=lambda x: x[0])
    if num_limit > 0:
        for name, param in kv_ls[:num_limit] + kv_ls[-num_limit:]:
            print("print_model[%s].Parameter %s = %s" % (desc, name, param))
    else:
        for name, param in kv_ls:
            print("print_model[%s].Parameter %s = %s" % (desc, name, param))


def track_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"【{tag}】Current Memory: {allocated:.2f} MB | max: {max_allocated:.2f} MB")


if __name__ == '__main__':

    def run_train():
        t1 = time.time()
        print("set default device to GPU")

        args = parse_args()
        for k, v in vars(args).items():
            print(f"{k}:{v}")

        # prepare data
        root_path = args.root_path
        
        prefix = root_path + "/data/"
        realshow_prefix = os.path.join(prefix, "all_stage")
        path_to_train_csv_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_csv_path = os.path.join(realshow_prefix, line.strip() + '.feather')
                path_to_train_csv_lst.append(tmp_csv_path)

        num_of_train_csv = len(path_to_train_csv_lst)
        print("training files:%s" % path_to_train_csv_lst)
        print(f"number of train_csv: {num_of_train_csv}")
        for idx, filepath in enumerate(path_to_train_csv_lst):
            print(f"{idx}: {filepath}")

        seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
        path_to_train_seq_pkl_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_seq_pkl_path = os.path.join(seq_prefix, line.strip() + '.pkl')
                path_to_train_seq_pkl_lst.append(tmp_seq_pkl_path)

        print("training seq files:")
        for idx, filepath in enumerate(path_to_train_seq_pkl_lst):
            print(f"{idx}: {filepath}")

        request_id_prefix = os.path.join(prefix, "request_id_dict")
        path_to_train_request_pkl_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_request_pkl_path = os.path.join(request_id_prefix, line.strip() + ".pkl")
                path_to_train_request_pkl_lst.append(tmp_request_pkl_path)

        print("training request files")
        for idx, filepath in enumerate(path_to_train_request_pkl_lst):
            print(f"{idx}: {filepath}")

        others_prefix = os.path.join(prefix, "others")
        path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
        print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")

        id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
        for k, v in id_cnt_dict.items():
            print(f"{k}:{v}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        prerank_model = DIN(
            args.emb_dim, args.seq_len,
            device, id_cnt_dict,
            nn_units=args.nn_units
        ).to(device)

        retrival_model = DSSM(
            args.emb_dim, args.seq_len,
            device, id_cnt_dict,
            nn_units=args.nn_units).to(device)

        prerank_optimizer = torch.optim.Adam(prerank_model.parameters(), lr=args.lr)
        retrival_optimizer = torch.optim.Adam(retrival_model.parameters(), lr=args.lr)

        if args.loss_type.startswith("lcron"):
            loss_model = LcronLossModel(device)
            loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=args.lr)
            print("Optimizer parameters:", [p.shape for p in loss_optimizer.param_groups[0]['params']])
        elif args.loss_type == "arf" or args.loss_type == "arf_v2":
            loss_model_prerank = ArfLossModel(device)
            loss_model_retrival = ArfLossModel(device)
            loss_params = itertools.chain(loss_model_prerank.parameters(), loss_model_retrival.parameters())
            loss_optimizer = torch.optim.Adam(loss_params, lr=args.lr)
        else:
            loss_optimizer = None

        num_workers,rank_offset = 1,0
        # train each model with just one epoch. epcoh is used to check the variance of metrics.
        for epoch in [args.epochs]:
            if args.epochs > 1:
                prefix = "/checkpoints/E%s" % (epoch)
            else:
                prefix = "/checkpoints/"
            for n_day in range(num_of_train_csv):
                print("TRAIN. processing n_day:%s" % (n_day))
                train_dataset = Rank_Train_All_BY_RANK_Dataset(
                    path_to_train_csv_lst[n_day],
                    args.seq_len,
                    path_to_train_seq_pkl_lst[n_day],
                    path_to_train_request_pkl_lst[n_day],
                    rank_offset=rank_offset
                )
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=True
                )

                print("{def} args.loss_type=%s" % args.loss_type)
                for iter_step, inputs in enumerate(train_loader):
                    inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:15]]
                    prerank_logits_list = prerank_model.forward_all_by_rank(inputs_LongTensor)
                    retrival_logits_list = retrival_model.forward_all_by_rank(inputs_LongTensor)
                    if iter_step % 100 == 0:
                        track_memory("BACKWARD_MEM")
                    if args.loss_type == "fs_ranknet": # RankNet
                        if iter_step == 1:
                            print("PRE loss_cal, mem_sum=%s" % torch.cuda.memory_summary())

                        mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]
                        loss, rank_pos_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss = compute_fsltr_loss_all_by_rank(
                            mask_list, prerank_logits_list, args, device,
                            is_debug=iter_step < 10
                        )
                        prerank_optimizer.zero_grad()
                        loss.backward()
                        prerank_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Prerank. loss={args.loss_type}. Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} "
                                f"\trank_pos_rank_loss:{rank_pos_rank_loss.detach().cpu().item():.4f} "
                                f"\trank_neg_rank_loss:{rank_neg_rank_loss.detach().cpu().item():.4f}"
                                f"\tcoarse_neg_rank_loss:{coarse_neg_rank_loss.detach().cpu().item():.4f}")

                        loss, rank_pos_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss = compute_fsltr_loss_all_by_rank(
                            mask_list, retrival_logits_list, args, device,
                        )
                        retrival_optimizer.zero_grad()
                        loss.backward()
                        retrival_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Retrival. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} "
                                f"\trank_pos_rank_loss:{rank_pos_rank_loss.detach().cpu().item():.4f} "
                                f"\trank_neg_rank_loss:{rank_neg_rank_loss.detach().cpu().item():.4f}"
                                f"\tcoarse_neg_rank_loss:{coarse_neg_rank_loss.detach().cpu().item():.4f}")
                    elif args.loss_type.startswith("bce"): # BCE loss
                        if iter_step == 1:
                            print("PRE loss_cal, mem_sum=%s" % torch.cuda.memory_summary())

                        prerank_loss, retrival_loss = compute_bce_metrics(
                                inputs, prerank_logits_list, retrival_logits_list, device)

                        prerank_optimizer.zero_grad()
                        prerank_loss.backward()
                        prerank_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Prerank. loss={args.loss_type}. Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{prerank_loss.detach().cpu().item():.4f} ")

                        retrival_optimizer.zero_grad()
                        retrival_loss.backward()
                        retrival_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Retrival. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{retrival_loss.detach().cpu().item():.4f} ")

                    elif args.loss_type == "icc":
                        outputs = compute_icc_metrics(inputs, prerank_logits_list, retrival_logits_list, device, joint_loss_conf)
                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        loss.backward()

                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")
                    
                    elif args.loss_type.startswith("rankflow"):
                        outputs = compute_rankflow_metrics(inputs, prerank_logits_list, retrival_logits_list,
                                                            device,
                                                            alpha=0,
                                                            is_debug=n_day < 5
                                                            )

                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        loss.backward()
                        
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

                    elif args.loss_type == "arf" or args.loss_type == "arf_v2":
                        outputs = compute_arf_metrics(inputs, prerank_logits_list, retrival_logits_list, device,
                                                      max_num=max_num, prerank_loss_conf=prerank_arf_loss_conf,
                                                      logger=logger,
                                                      retrival_loss_conf=retrival_arf_loss_conf,
                                                      loss_model_prerank=loss_model_prerank,
                                                      loss_model_retrival=loss_model_retrival,
                                                      loss_type=args.loss_type)
                        prerank_loss = outputs["prerank_loss"]
                        retrival_loss = outputs["retrival_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        prerank_loss.backward()
                        retrival_loss.backward()

                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tprerank_loss:{prerank_loss.detach().cpu().item():.4f}\tretrival_loss:{retrival_loss.detach().cpu().item():.4f} ")

                    elif args.loss_type.startswith("lcron"):
                        if len(args.loss_type.split("_")) > 1:
                            version = args.loss_type.split("_")[1]
                        else:
                            version = 'v0'
                        joint_loss_conf.version = version
                        outputs = compute_lcron_metrics(inputs, prerank_logits_list, retrival_logits_list, device,
                                                        max_num=max_num,
                                                        joint_loss_conf=joint_loss_conf,
                                                        logger=logger,
                                                        loss_model=loss_model,
                                                        sort='neural_sort')
                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        loss_optimizer.zero_grad()
                        loss.backward()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        loss_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")
                            
                    elif args.loss_type.startswith("fs_lambdaloss"):
                        outputs = compute_fsltr_lambda_all_by_rank(inputs, prerank_logits_list, retrival_logits_list,
                                                                   device,
                                                                   smooth_fraction=0.8,
                                                                   topn=30,
                                                                   is_debug=n_day < 5
                                                                   )
                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()
                        loss.backward()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

        path_to_save_model = root_path + prefix + f"prerank_tau-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}.pkl"
        torch.save(prerank_model.state_dict(), path_to_save_model)
        print("Saving prerank model to path_to_save_model:%s" % path_to_save_model)
        path_to_save_model = root_path + prefix + f"retrival_tau-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}.pkl"
        torch.save(retrival_model.state_dict(), path_to_save_model)
        print("Saving retrival model to path_to_save_model:%s" % path_to_save_model)

        t2 = time.time()
        print("time_used:%s" % (t2 - t1))

    run_train()
