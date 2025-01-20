export CUDA_VISIBLE_DEVICES=1

model_name=TimeFilter
seq_len=96

for pred_len in 96
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path solar_AL.txt \
    --model_id solar_$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --dropout 0.2 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1
done

for pred_len in 192 336
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path solar_AL.txt \
    --model_id solar_$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --dropout 0.3 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1
done

for pred_len in 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path solar_AL.txt \
    --model_id solar_$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --dropout 0.6 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1
done
