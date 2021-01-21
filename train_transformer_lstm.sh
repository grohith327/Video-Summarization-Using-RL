for seed in {1..5}
do
    python trainer.py --seed $seed  --run_name transformer_lstm_seed_$seed --decoder transformer --dataset tvsum
done