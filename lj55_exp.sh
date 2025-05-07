for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0, python examples/lj55_fab.py training.seed=$seed training.resume=False # Training
    CUDA_VISIBLE_DEVICES=0, python examples/lj55_fab.py training.seed=$seed training.resume=True # Sampling
done