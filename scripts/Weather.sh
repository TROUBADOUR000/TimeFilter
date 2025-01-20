export CUDA_VISIBLE_DEVICES=3

model_name=TimeFilter

# 96
seq_len=96

for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 2 \
    --root_path ./data \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1
done

# long horizon
seq_len=720

for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --patch_len 144 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --train_epochs 10 \
    --dropout 0.6 \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --itr 1
done
