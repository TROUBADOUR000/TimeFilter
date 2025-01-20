export CUDA_VISIBLE_DEVICES=7

model_name=TimeFilter
seq_len=96

for pred_len in 12 24 48
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path PEMS08.npz \
  --model_id PEMS08_$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --patch_len 48 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 20 \
  --itr 1 \
  --use_norm 1
done
