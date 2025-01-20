export CUDA_VISIBLE_DEVICES=7

model_name=TimeFilter
seq_len=720

for pred_len in 96 192
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path climate.csv \
    --model_id climate_$seq_len'_'$pred_len \
    --model $model_name \
    --data Climate \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1763 \
    --dec_in 1763 \
    --c_out 1763 \
    --patch_len 720 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --train_epochs 10 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --itr 1
done

for pred_len in 336
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path climate.csv \
    --model_id climate_$seq_len'_'$pred_len \
    --model $model_name \
    --data Climate \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1763 \
    --dec_in 1763 \
    --c_out 1763 \
    --patch_len 720 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --train_epochs 10 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --itr 1
done

for pred_len in 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path climate.csv \
    --model_id climate_$seq_len'_'$pred_len \
    --model $model_name \
    --data Climate \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1763 \
    --dec_in 1763 \
    --c_out 1763 \
    --patch_len 720 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --train_epochs 10 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --itr 1
done