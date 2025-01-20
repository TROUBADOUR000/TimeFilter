export CUDA_VISIBLE_DEVICES=7

model_name=DUET
seq_len=48

for pred_len in 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../TimeFilter/data/ \
    --data_path climate.csv \
    --model_id climate_$seq_len'_'$pred_len \
    --model $model_name \
    --data Climate \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 1 \
    --enc_in 1763 \
    --dec_in 1763 \
    --c_out 1763 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --patch_len 48 \
    --n_heads 4 \
    --train_epochs 10 \
    --dropout 0.1 \
    --batch_size 16 \
    --itr 1
done
