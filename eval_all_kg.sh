DEVICE=$1
DATA_ROOT=./data/
LOGPATH=./logs
PATH_TO_CHECKPOINT=./state/Wiki_PT_GraphPrompter/state_dict
SHOTS=10 # where SHOTS means the size of candidate prompt set
WAYS=20
TEMP=3 # where TEMP means is the number of selected prompts from the candidate prompt set

LOGFILE=kg.log

echo "-----use ${PATH_TO_CHECKPOINT}-----" >> $LOGFILE


# DATASET: ConceptNet
echo "-----concepnet-----" >> $LOGFILE
python3 experiments/run_single_experiment.py --root $DATA_ROOT  --dataset ConceptNet --emb_dim 256 -shot $SHOTS --temp $TEMP --device $DEVICE --input_dim 768 --layers E2,UX,M2  -ds_cap 10 --prefix InContext_eval_PRODIGY -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way 4 -bs 1 -qry 4 --ignore_label_embeddings True --task multiway_classification -test_cap 500 -val_cap 10 --workers 15  --eval -meta_pos  True  --eval_only True  -pretrained $PATH_TO_CHECKPOINT --all_test True  --knn --select --eval --no_split_labels True  --label_set 0 1 2 3 4 5 6 7 8 9 10 11 12 13 2>&1 | tee  > $LOGPATH/eval_conceptNet.log
grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_conceptNet.log   >> $LOGFILE


# DATASET: FB15K-237
echo "-----fb15k-237-----" >> $LOGFILE
python3 experiments/run_single_experiment.py --root $DATA_ROOT  --dataset FB15K-237 --emb_dim 256 -shot $SHOTS --temp $TEMP --device $DEVICE --input_dim 768 --layers E2,UX,M2  -ds_cap 10 --prefix InContext_eval_PRODIGY -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way $WAYS -bs 1 -qry 4 --ignore_label_embeddings True --task multiway_classification -test_cap 500 -val_cap 10 --workers 15  --eval -meta_pos  True --eval_only True  -pretrained $PATH_TO_CHECKPOINT --all_test True --cache --knn  --eval --cache_cap 3 --select --no_split_labels True  --label_set 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 2>&1 | tee   $LOGPATH/eval_fb15k-237_w${WAYS}_knnsl.log
grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_fb15k-237_w${WAYS}_knnsl.log  >> $LOGFILE


# DATASET: NELL
echo "-----nell-----" >> $LOGFILE
python3 experiments/run_single_experiment.py --root $DATA_ROOT  --dataset NELL --emb_dim 256 -shot $SHOTS --temp $TEMP --device $DEVICE --input_dim 768 --layers E2,UX,M2  -ds_cap 10 --prefix InContext_eval_PRODIGY -esp 500 --eval_step 1000 --epochs 1 --dropout 0 --n_way $WAYS -bs 1 -qry 4 --ignore_label_embeddings True --task multiway_classification -test_cap 500 -val_cap 10 --workers 15  -meta_pos  True  --eval_only True  -pretrained $PATH_TO_CHECKPOINT --all_test True --select --knn --cache --eval --no_split_labels True --cache_cap 3 --label_set 1 2 3 4 6 9 10 11 13 14 16 18 19 22 25 27 29 31 32 35 38 39 42 45 46 51 55 57 59 60 62 63 66 69 70 71 73 77 78 79 82 84 88 90 91 92 93 94 102 105 106 107 108 109 110 112 115 116 120 121 122 123 126 127 128 129 130 136 143 152 155 157 158 159 160 168 170 171 173 175 176 177 178 181 183 186 187 188 189 190 192 194 195 198 202 204 206 209 212 215 217 219 220 221 225 227 230 231 236 240 253 255 256 257 258 259 261 262 263 264 266 267 273 274 276 277 279 281 282 283 285 286 289 2902>&1 | tee $LOGPATH/eval_nell_ada_t3w${WAYS}_knn_ul.log 
grep -E '^round|^wandb:     start_test_acc|^wandb: start_test_acc_std' $LOGPATH/eval_nell_ada_t3w${WAYS}_knn_ul.log  >> $LOGFILE