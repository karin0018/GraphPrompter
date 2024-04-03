DEVICE=$1
DATA_ROOT=./data
LOGPATH=./logs

python experiments/run_single_experiment.py --dataset mag240m  --root $DATA_ROOT  --original_features True -ds_cap 50010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 30 -shot 3 -qry 4 -eval_step 1000 -task cls_nm_sb  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device $DEVICE --prefix MAG_PT_GraphPrompter --contras_loss  2>&1 | tee $LOGPATH/pretrain_GraphPrompter_mag240m.log

