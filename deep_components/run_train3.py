import itertools
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from models import DSSM, DIN, MLP
from dataset3 import Rank_Train_All_BY_RERANK_Dataset
from deep_components.loss.three_stage.lcron import LcronLossModel
from loss.three_stage.arf import ArfLossModel
from utils import load_pkl

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

### GLOBAL
# rerank_pos(10) + rerank_neg(10) + rank_neg(10) + coarse_neg(10) + prerank_neg(10)
max_num = 50

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
    parser.add_argument('--tau', type=float, default=1, help='tau.')
    parser.add_argument('--root_path', type=str, default=".", help='root path to data, checkpoints and logs')
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
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)

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

        rank_model = DIN(
            args.emb_dim, args.seq_len,
            device, id_cnt_dict,
            nn_units=args.nn_units
        ).to(device)

        prerank_model = MLP(
            args.emb_dim, args.seq_len,
            device, id_cnt_dict,
            nn_units=args.nn_units
        ).to(device)

        retrival_model = DSSM(
            args.emb_dim, args.seq_len,
            device, id_cnt_dict,
            nn_units=args.nn_units).to(device)

        rank_optimizer = torch.optim.Adam(rank_model.parameters(), lr=args.lr)
        prerank_optimizer = torch.optim.Adam(prerank_model.parameters(), lr=args.lr)
        retrival_optimizer = torch.optim.Adam(retrival_model.parameters(), lr=args.lr)

        if args.loss_type.startswith("lcron"):
            loss_model = LcronLossModel(device)
            loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=args.lr)
            print("Optimizer parameters:", [p.shape for p in loss_optimizer.param_groups[0]['params']])
        elif args.loss_type == "arf" or args.loss_type == "arf_v2":
            loss_model_rank = ArfLossModel(device)
            loss_model_prerank = ArfLossModel(device)
            loss_model_retrival = ArfLossModel(device)
            loss_params = itertools.chain(loss_model_rank.parameters(), loss_model_prerank.parameters(),
                                          loss_model_retrival.parameters())
            loss_optimizer = torch.optim.Adam(loss_params, lr=args.lr)
        else:
            loss_optimizer = None

        num_workers, rank_offset = 1, 0
        # train each model with just one epoch. epcoh is used to check the variance of metrics.
        for epoch in [args.epochs]:
            if args.epochs > 1:
                prefix = "/checkpoints/E%s" % (epoch)
            else:
                prefix = "/checkpoints/"
            for n_day in range(num_of_train_csv):
                print("TRAIN. processing n_day:%s" % (n_day))
                train_dataset = Rank_Train_All_BY_RERANK_Dataset(
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

                is_debug = False
                print("{def} args.loss_type=%s" % args.loss_type)
                print("{def} args.is_debug=%s" % is_debug)
                for iter_step, inputs in enumerate(train_loader):
                    if iter_step == 1:
                        for k, inp in enumerate(inputs):
                            print("DEBUG_SHAPE. iter=%s, inp[%s]=%s" % (iter_step, k, inp.shape))
                        for k, inp in enumerate(inputs):
                            print("DEBUG_DATA. iter=%s, inp[%s]=%s" % (iter_step, k, inp))
                    if is_debug and iter_step > 1000:
                        print("EARLIY STOP, FOR DEBUG ONLY")
                        break
                    inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:16]]
                    rank_logits_list = rank_model.forward_all_by_rerank(inputs_LongTensor)
                    prerank_logits_list = prerank_model.forward_all_by_rerank(inputs_LongTensor)
                    retrival_logits_list = retrival_model.forward_all_by_rerank(inputs_LongTensor)
                    if iter_step % 100 == 0:
                        track_memory("BACKWARD_MEM")

                    if args.loss_type == "fs_ranknet":
                        if iter_step == 1:
                            print("PRE loss_cal, mem_sum=%s" % torch.cuda.memory_summary())
                        from loss.three_stage.fs_ranknet import compute_fs_ranknet_loss
                        mask_list = [tensor.to(device) for tensor in inputs[-10:-5]]

                        loss, rerank_pos_rank_loss, rerank_neg_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss = compute_fs_ranknet_loss(
                            mask_list, rank_logits_list, device, is_debug=iter_step < 10
                        )
                        rank_optimizer.zero_grad()
                        loss.backward()
                        rank_optimizer.step()

                        loss, rerank_pos_rank_loss, rerank_neg_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss = compute_fs_ranknet_loss(
                            mask_list, prerank_logits_list, device, is_debug=iter_step < 10
                        )
                        prerank_optimizer.zero_grad()
                        loss.backward()
                        prerank_optimizer.step()

                        loss, rerank_pos_rank_loss, rerank_neg_rank_loss, rank_neg_rank_loss, coarse_neg_rank_loss = compute_fs_ranknet_loss(
                            mask_list, retrival_logits_list, device, is_debug=iter_step < 10
                        )
                        retrival_optimizer.zero_grad()
                        loss.backward()
                        retrival_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Retrival. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} "
                                f"\trerank_pos_rank_loss:{rerank_pos_rank_loss.detach().cpu().item():.4f} "
                                f"\trerank_neg_rank_loss:{rerank_neg_rank_loss.detach().cpu().item():.4f} "
                                f"\trank_neg_rank_loss:{rank_neg_rank_loss.detach().cpu().item():.4f}"
                                f"\tcoarse_neg_rank_loss:{coarse_neg_rank_loss.detach().cpu().item():.4f}")

                    elif args.loss_type.startswith("bce"):
                        if iter_step == 1:
                            print("PRE loss_cal, mem_sum=%s" % torch.cuda.memory_summary())
                        from loss.three_stage.bce import compute_bce_loss

                        rank_loss, prerank_loss, retrival_loss = compute_bce_loss(
                            inputs, rank_logits_list, prerank_logits_list, retrival_logits_list, device)

                        rank_optimizer.zero_grad()
                        rank_loss.backward()
                        rank_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Prerank. loss={args.loss_type}. Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{rank_loss.detach().cpu().item():.4f} ")

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
                        from loss.three_stage.icc import compute_icc_loss
                        outputs = compute_icc_loss(inputs, rank_logits_list, prerank_logits_list,
                                                      retrival_logits_list, device)
                        loss = outputs["total_loss"]
                        rank_optimizer.zero_grad()
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        loss.backward()

                        rank_optimizer.step()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

                    elif args.loss_type.startswith("rankflow"):
                        from loss.three_stage.rankflow import compute_rankflow_loss
                        outputs = compute_rankflow_loss(inputs, rank_logits_list, prerank_logits_list,
                                                           retrival_logits_list, device)
                        loss = outputs["total_loss"]
                        rank_optimizer.zero_grad()
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        loss.backward()

                        rank_optimizer.step()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

                    elif args.loss_type == "arf" or args.loss_type == "arf_v2":
                        from loss.three_stage.arf import compute_arf_loss
                        outputs = compute_arf_loss(inputs, rank_logits_list, prerank_logits_list,
                                                      retrival_logits_list, device,
                                                      loss_model_rank=loss_model_rank,
                                                      loss_model_prerank=loss_model_prerank,
                                                      loss_model_retrival=loss_model_retrival,
                                                      loss_type=args.loss_type)
                        rank_loss = outputs["rank_loss"]
                        prerank_loss = outputs["prerank_loss"]
                        retrival_loss = outputs["retrival_loss"]

                        rank_optimizer.zero_grad()
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        rank_loss.backward()
                        prerank_loss.backward()
                        retrival_loss.backward()

                        rank_optimizer.step()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\trank_loss:{rank_loss.detach().cpu().item():.4f}\tprerank_loss:{prerank_loss.detach().cpu().item():.4f}\tretrival_loss:{retrival_loss.detach().cpu().item():.4f} ")

                    elif args.loss_type.startswith("lcron"):
                        if len(args.loss_type.split("_")) > 1:
                            version = args.loss_type.split("_")[1]
                        else:
                            version = 'v0'
                        from loss.three_stage.lcron import compute_lcron_loss
                        outputs = compute_lcron_loss(inputs, rank_logits_list, prerank_logits_list,
                                                        retrival_logits_list, device,
                                                        loss_model, version)
                        loss = outputs["total_loss"]
                        rank_optimizer.zero_grad()
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        loss_optimizer.zero_grad()

                        loss.backward()
                        rank_optimizer.step()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

                    elif args.loss_type.startswith("fs_lambdaloss"):
                        from loss.three_stage.fs_lambdaloss import compute_fs_lambdaloss
                        outputs = compute_fs_lambdaloss(inputs, rank_logits_list, prerank_logits_list,
                                                                   retrival_logits_list, device, 0.8, 30)
                        loss = outputs["total_loss"]
                        rank_optimizer.zero_grad()
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        if loss_optimizer: loss_optimizer.zero_grad()

                        loss.backward()
                        rank_optimizer.step()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        if loss_optimizer: loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")

        path_to_save_model = root_path + prefix + f"rank_tau-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}_S3.pkl"
        torch.save(rank_model.state_dict(), path_to_save_model)
        print("Saving rank model to path_to_save_model:%s" % path_to_save_model)

        path_to_save_model = root_path + prefix + f"prerank_tau-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}_S3.pkl"
        torch.save(prerank_model.state_dict(), path_to_save_model)
        print("Saving prerank model to path_to_save_model:%s" % path_to_save_model)

        path_to_save_model = root_path + prefix + f"retrival_tau-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}_S3.pkl"
        torch.save(retrival_model.state_dict(), path_to_save_model)
        print("Saving retrival model to path_to_save_model:%s" % path_to_save_model)

        t2 = time.time()
        print("time_used:%s" % (t2 - t1))


    run_train()
