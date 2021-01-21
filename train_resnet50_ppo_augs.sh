for seed in {1..5}
do
    python trainer.py --seed $seed  --run_name resnet50_augs_PPO_seed_$seed --cnn_feat resnet50 --dataset tvsum --algorithm ppo --use_augs --epochs 80
done