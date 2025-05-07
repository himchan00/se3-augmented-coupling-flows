for seed in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0, python examples/dw4_fab.py training.seed=$seed training.resume=False # Training
    CUDA_VISIBLE_DEVICES=0, python examples/dw4_fab.py training.seed=$seed training.resume=True # Sampling
done