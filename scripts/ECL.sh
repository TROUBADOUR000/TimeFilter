export CUDA_VISIBLE_DEVICES=5

model_name=TimeFilter

# 96
seq_len=96

for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --patch_len 32 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 15 \
  --d_model 512\
  --d_ff 512\
  --dropout 0.5 \
  --itr 1
done

for pred_len in 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --patch_len 32 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 15 \
  --d_model 512\
  --d_ff 512\
  --dropout 0.4 \
  --itr 1
done


# long horizon
seq_len=512

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --patch_len 128 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 15 \
  --d_model 512\
  --d_ff 512\
  --dropout 0.5 \
  --top_p 0.0 \
  --itr 1
done
