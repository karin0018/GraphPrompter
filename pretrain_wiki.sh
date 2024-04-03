DEVICE=$1
DATA_ROOT=./data
LOGPATH=./logs

python3 experiments/run_single_experiment.py --root $DATA_ROOT --dataset Wiki --emb_dim 256 --device $DEVICE --input_dim 768 --workers 7 --layers E2,UX,M2 -ds_cap 20010 -lr 1e-3 --prefix Wiki_PT_GraphPrompter -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 15 -bs 1 -qry 4 -shot 3 --task cls_nm --ignore_label_embeddings True -val_cap 10 -test_cap 10 -aug ND0.5,NZ0.5 -aug_test True -ckpt_step 1000 --all_test True -attr 1000 -meta_pos True --select 2>&1 | tee $LOGPATH/pretrain_GraphPrompter_Wiki.log

