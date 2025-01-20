export CUDA_VISIBLE_DEVICES=3

model_name=TimeFilter
seq_len=96

for pred_len in 96 192 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.8 \
  --patch_len 2 \
  --pos 0 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1
done

for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.8 \
  --patch_len 2 \
  --pos 0 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 128 \
  --d_ff 128 \
  --itr 1
done
