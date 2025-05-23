from typing import Callable, Tuple, Optional, NamedTuple, Any

import chex
import jax
import numpy as np
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import time
import pathlib
import optax
import wandb

from eacf.train.base import get_leading_axis_tree
from eacf.utils.plotting import plot_history
from eacf.utils.loggers import Logger, ListLogger, WandbLogger
from eacf.utils.checkpoints import get_latest_checkpoint


class TrainingState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    key: chex.PRNGKey


InitStateFn = Callable[[TrainingState], TrainingState]
UpdateStateFn = Callable[[TrainingState], Tuple[TrainingState, dict]]
EvalAndPlotFn = Callable[[TrainingState, chex.PRNGKey, int, bool, str], dict]


class TrainConfig(NamedTuple):
    n_iteration: int
    logger: Logger
    seed: int
    n_checkpoints: int
    n_eval: int
    init_state: InitStateFn
    update_state: UpdateStateFn
    eval_and_plot_fn: EvalAndPlotFn
    save: bool = True
    save_dir: str = "/tmp"
    resume: bool = False
    wandb_upload_each_time: bool = True
    use_64_bit: bool = False
    runtime_limit: Optional[float] = None
    save_state_all_devices: bool = False


def train(config: TrainConfig):
    """Generic Training script."""
    if config.runtime_limit:
        start_time = time.time()

    if config.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    if config.save:
        pathlib.Path(config.save_dir).mkdir(exist_ok=True)  # base saving directory

        plots_dir = os.path.join(config.save_dir, f"plots")
        pathlib.Path(plots_dir).mkdir(exist_ok=config.resume)

        checkpoints_dir = os.path.join(config.save_dir, f"model_checkpoints")
        pathlib.Path(checkpoints_dir).mkdir(exist_ok=config.resume)
    else:
        plots_dir = None
        checkpoints_dir = None

    checkpoint_iter_np = np.flip(
        np.linspace(
            config.n_iteration - 1, 0, config.n_checkpoints, dtype="int", endpoint=False
        )
    )
    checkpoint_iter = list(checkpoint_iter_np)
    eval_iter = list(
        np.flip(
            np.linspace(
                config.n_iteration - 1, 0, config.n_eval, dtype="int", endpoint=False
            )
        )
    )

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)

    state = config.init_state(subkey)

    start_iter = 0
    if config.resume:
        latest_cp = get_latest_checkpoint(checkpoints_dir, key="state_")
        if latest_cp:
            with open(latest_cp, "rb") as f:
                state = pickle.load(f)
            print(f"loaded checkpoint {latest_cp}")
        else:
            print("no checkpoint found, starting training from scratch")

    # Run eval and plot before training starts
    if start_iter == 0:
        key, subkey = jax.random.split(key)
        eval_info = config.eval_and_plot_fn(state, subkey, -1, config.save, plots_dir)
        if config.resume:
            return # early exit after sample generation
        eval_info.update(iteration=-1)
        config.logger.write(eval_info)
        print(f"initial model eval complete, eval info: \n {eval_info}")

    pbar = tqdm(range(start_iter, config.n_iteration))
    tvd_best = np.inf
    for iteration in pbar:
        state, info = config.update_state(state)

        # check for scalar info -- usually if last batch info is active
        leading_info_shape = get_leading_axis_tree(info, 1)
        if len(leading_info_shape) == 0 or leading_info_shape == (1,):
            info.update(iteration=iteration)
            config.logger.write(info)

        else:
            for batch_idx in range(leading_info_shape[0]):
                batch_info = jax.tree_util.tree_map(lambda x: x[batch_idx], info)
                batch_info.update(iteration=iteration)
                config.logger.write(batch_info)

        if config.eval_and_plot_fn is not None and iteration in eval_iter:
            key, subkey = jax.random.split(key)
            eval_info = config.eval_and_plot_fn(
                state, subkey, iteration, config.save, plots_dir
            )
            eval_info.update(iteration=iteration)
            pbar.write(str(eval_info))
            config.logger.write(eval_info)
            tvd = eval_info["tvd"]
            if tvd < tvd_best:
                tvd_best = tvd
                print(f"new best model with tvd: {tvd_best}")
                checkpoint_path = os.path.join(
                    checkpoints_dir, "state_%08i.pkl" % iteration
                )
                with open(checkpoint_path, "wb") as f:
                        pickle.dump(state, f)
                        print(f"saved checkpoint to {checkpoint_path}")
                with open(checkpoints_dir + "/best_model.txt", "a") as f:
                    f.write(f"iteration: {iteration}, tvd: {tvd}\n")

    if isinstance(config.logger, ListLogger):
        plot_history(config.logger.history)
        plt.show()

    if isinstance(config.logger, WandbLogger) and config.save:
        wandb.save(str(pathlib.Path(checkpoints_dir)) + "/*",  base_path=config.save_dir, policy="now")
        wandb.save(str(pathlib.Path(plots_dir)) + "/*", base_path=config.save_dir, policy="now")

    return config.logger, state
