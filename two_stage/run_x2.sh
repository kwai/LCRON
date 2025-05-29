loss_type=$2
cuda=$3
if [ -n "$4" ]; then
    tau=$4
else
    tau="1.0"
fi
if [ -n "$5" ]; then
    epochs=$5
else
    epochs="0"
fi
if [ -n "$6" ]; then
    root_path=$6
else
    root_path="."
fi
if [ -n "$7" ]; then
    lr=$7
else
    lr=1e-2
fi
if [ -n "$8" ]; then
    BS=$8
else
    BS=1024
fi


# 输出结果
echo "tau=${tau}"
echo "epochs=${epochs}"
echo "lr=${lr}"
echo "BS=${BS}"
echo "epochs=${epochs}"
echo "root_path=${root_path}"

tag=${loss_type}-1st

for epoch in $epochs; do
  echo "epoch=${epoch}"
  TRAIN_PATH="${root_path}/logs/TRAIN_bs-${BS}_lr-${lr}_tau${tau}_${tag}_E${epoch}_S2.log"
  TEST_PATH="${root_path}/logs/TEST_bs-${BS}_lr-${lr}_tau${tau}_${tag}_E${epoch}_S2.log"

  if [[ "$1" == "all" || "$1" == "train" ]]; then
    python -B -u deep_components/run_train2.py \
    --epochs=${epoch} \
    --loss_type=${loss_type} \
    --tau="${tau}" \
    --batch_size=${BS} \
    --infer_realshow_batch_size=${BS} \
    --infer_recall_batch_size=${BS} \
    --emb_dim=8 \
    --lr=${lr} \
    --seq_len=50 \
    --cuda=${cuda} \
    --root_path=${root_path} \
    --print_freq=100 \
    --tag=${tag} > $TRAIN_PATH 2>&1
  fi

  if [[ "$1" == "all" || "$1" == "test" ]];
   then
    python -B -u deep_components/run_test2.py \
    --epochs=${epoch} \
    --loss_type=${loss_type} \
    --tau="${tau}" \
    --batch_size=${BS} \
    --infer_realshow_batch_size=${BS} \
    --infer_recall_batch_size=${BS} \
    --emb_dim=8 \
    --lr=${lr} \
    --seq_len=50 \
    --cuda=${cuda} \
    --root_path=${root_path} \
    --print_freq=100 \
    --tag=${tag} > $TEST_PATH 2>&1

    # collect metrics
    sh two_stage/run_collect.sh ${loss_type}

  fi

done
