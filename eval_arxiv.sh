DEVICE=$1
DATA_ROOT=$2
LOGPATH=./logs
PATH_TO_CHECKPOINT=./state/MAG_PT_GraphPrompter/state_dict
# ways=5
shots=10
temp=3

LOGFILE=arxiv.log

echo "-----use ${PATH_TO_CHECKPOINT}-----" >> $LOGFILE

# original shot=3 means user 3 prompts
for ways in 3 5 10 20 40;
do
echo "ways = ${ways}" >> $LOGFILE
# echo "---no knn----" >> $LOGFILE

echo "----- original -----" >> $LOGFILE
# original shot=3 means user 3 prompts
python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot 3 -bs 1 -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee > $LOGPATH/eval_arxiv_s3w${ways}.log

grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_arxiv_s3w${ways}.log   >> $LOGFILE

echo "----- knn -----" >> $LOGFILE
# use knn shot=10 temp=3 means choice 3 prompts form 10 candidates
python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot $shots --temp $temp -bs 1  -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --knn --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee $LOGPATH/eval_arxiv_t${temp}w${ways}_knn.log

grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_arxiv_t${temp}w${ways}_knn.log   >> $LOGFILE

echo "----- knn cache-----" >> $LOGFILE
python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot $shots --temp $temp -bs 1  -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --knn --cache --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee $LOGPATH/eval_arxiv_t${temp}w${ways}_knn_es.log

grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_arxiv_t${temp}w${ways}_knn_es.log   >> $LOGFILE


done


# # use knn shot=10 temp=3 means choice 3 prompts form 10 candidates
# python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot $shots --temp $temp -bs 1  -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --knn --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee $LOGPATH/eval_arxiv_t${temp}w${ways}_knn.log

# # use edge_sampler
# python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot $shots --temp $temp -bs 1  -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --edge_sampler --edge_sampler_type degree --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee $LOGPATH/eval_arxiv_t${temp}w${ways}_es.log

# use knn & edge_sampler
# python experiments/run_single_experiment.py --dataset arxiv --root $DATA_ROOT  -ds_cap 510 -val_cap 510 -test_cap 500 -eval_step 100 -epochs 1 --layers S2,U,M -way $ways -shot $shots --temp $temp -bs 1  -qry 3 -lr 1e-5 -bert roberta-base-nli-stsb-mean-tokens -pretrained $PATH_TO_CHECKPOINT --eval_only True --train_cap 10 --seed 10 --knn --edge_sampler --edge_sampler_type degree --eval --emb_dim 256 --device $DEVICE --input_dim 768 2>&1 | tee $LOGPATH/eval_arxiv_t${temp}w${ways}_knn_es_maskedge.log

