from typing import Callable, Optional, List, Tuple, Union

from functools import partial
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import os
from eacf.utils.plotting import bin_samples_by_dist

from eacf.train.fab_train_no_buffer import SequentialMonteCarloSampler, build_smc_forward_pass, LogProbFn, TrainStateNoBuffer
from eacf.train.fab_train_with_buffer import TrainStateWithBuffer
from eacf.flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams


mpl.rcParams['figure.dpi'] = 150
PlottingBatchSize = int
TrainData = FullGraphSample
TestData = FullGraphSample
FLowPlotter = Callable[[AugmentedFlowParams, AugmentedFlow, chex.PRNGKey, PlottingBatchSize,
                        TestData, TrainData], List[plt.Figure]]


def process_samples(joint_samples: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    x, a = jnp.split(joint_samples, [1, ], axis=-2)
    a_minus_x = a - x
    x = jnp.squeeze(x, axis=-2)
    return x, a, a_minus_x


def plot_histogram_from_counts(count_list, bins, axs, labels):
    for i, counts in enumerate(count_list[:-1]):
        axs[i].stairs(counts, bins, alpha=0.4, fill=True, label=labels[i])
        axs[i].stairs(count_list[-1], bins, alpha=0.4, fill=True, label=labels[-1])
        axs[i].legend()


def make_default_plotter(
        test_data: FullGraphSample,
        flow: AugmentedFlow,
        ais: SequentialMonteCarloSampler,
        log_p_x: LogProbFn,
        n_samples_from_flow: int,
        max_n_samples: int = 10000,
        max_distance: Optional[float] = 20.,
):

    n_samples = n_samples_from_flow

    ais_forward = build_smc_forward_pass(flow, log_p_x, test_data.features[0], ais, n_samples)

    @partial(jax.jit)
    def get_data_for_plotting(state: Union[TrainStateNoBuffer, TrainStateWithBuffer],
                              key: chex.PRNGKey, test_data=test_data[:max_n_samples]):
        pos_x_target = test_data.positions
        target_batch = pos_x_target.shape[0]
        params = state.params
        key1, key2 = jax.random.split(key)
        # Get samples from flow and AIS.
        # target_batch_size == gen_batch_size
        assert target_batch % n_samples == 0
        n_gen = target_batch // n_samples
        l_pos_x_ais = []
        for i in range(n_gen):
            flow_samples, ais_samples, log_w = ais_forward(state.params, state.smc_state, key1)[:3]
            # Process samples.
            pos_x_flow, pos_a_flow, a_min_x_flow = process_samples(flow_samples)
            pos_x_ais, pos_a_ais, a_min_x_ais = process_samples(ais_samples)
            pos_a_target = flow.aux_target_sample_n_apply(params.aux_target, test_data, key2)
            a_min_x_target = pos_a_target - pos_x_target[:, :, None]
            l_pos_x_ais.append(pos_x_ais)
        pos_x_ais = jnp.concatenate(l_pos_x_ais, axis=0)

        # Get bins and counts
        bins_x, count_list_x = bin_samples_by_dist([pos_x_flow, pos_x_ais, pos_x_target], max_distance)
        bins_a, count_list_a = jax.vmap(bin_samples_by_dist, in_axes=(-2, None), out_axes=0
                                        )([pos_a_flow, pos_a_ais, pos_a_target], max_distance)
        bins_a_minus_x, count_list_a_minus_x = jax.vmap(bin_samples_by_dist, in_axes=(-2, None), out_axes=0)(
            [a_min_x_flow, a_min_x_ais, a_min_x_target], max_distance)
        return bins_x, count_list_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x, pos_x_ais, pos_x_target


    def default_plotter(state: TrainStateNoBuffer, key: chex.PRNGKey) -> dict:
        labels = ['flow', 'ais', 'target_energy']
        n_plots = len(labels) - 1

        # Plot interatomic distance histograms.
        bins_x, count_list_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x, gen_sample, gt_sample = \
            get_data_for_plotting(state, key)

        gen_sample = torch.from_numpy(np.array(gen_sample))
        gt_sample = torch.from_numpy(np.array(gt_sample))
        print(f'gen_sample shape: {gen_sample.shape}')
        print(f'gt_sample shape: {gt_sample.shape}')
        # Heuristic to detect sample generation mode (not training)
        if len(gen_sample) == 10000:
            i = 0
            while os.path.exists(f'gen_sample_{gen_sample.shape}_{i}.pt'):
                i += 1
                # Increment the file name if it already exists
            torch.save(gen_sample, f'gen_sample_{gen_sample.shape}_{i}.pt')
        # Compute the pairwise interatomic distances
        tvd = Atomic_TVD_particle(gen_sample, gt_sample)
        print(f'Total variation distance: {tvd}')

        # Plot original coords
        fig1, axs = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        plot_histogram_from_counts(count_list_x, bins_x, axs, labels)
        fig1.suptitle('x samples', fontsize=16)
        fig1.tight_layout()

        # Augmented info.
        fig2, axs = plt.subplots(flow.n_augmented, n_plots, figsize=(5*n_plots, 5*flow.n_augmented))
        axs = axs[None] if flow.n_augmented == 1 else axs
        for i in range(flow.n_augmented):
            counts = jax.tree_util.tree_map(lambda x: x[i], count_list_a)
            plot_histogram_from_counts(counts, bins_a[i], axs[i, :], labels)
        fig2.suptitle('a samples', fontsize=16)
        fig2.tight_layout()

        # Plot histogram (x - a)
        fig3, axs = plt.subplots(flow.n_augmented, n_plots, figsize=(5 * n_plots, 5*flow.n_augmented))
        axs = axs[None] if flow.n_augmented == 1 else axs
        for i in range(flow.n_augmented):
            counts = jax.tree_util.tree_map(lambda x: x[i], count_list_a_minus_x)
            plot_histogram_from_counts(counts, bins_a_minus_x[i], axs[i, :], labels)
        fig3.suptitle('a - x samples', fontsize=16)
        fig3.tight_layout()

        return [fig1, fig2, fig3], {'tvd': tvd}

    return default_plotter


# Added for evaluation

def Atomic_TVD_particle(gen_sample, gt_sample):
    gt_interatomic = interatomic_dist(gt_sample).detach().cpu()
    gen_interatomic = interatomic_dist(gen_sample).detach().cpu()
    return total_variation_distance(gen_interatomic, gt_interatomic)


def interatomic_dist(x):
    batchsize, n_particles, n_dim = x.shape
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,]
    dist = torch.linalg.norm(distances, dim=-1)
    return dist


def total_variation_distance(samples1, samples_test, bins=200):
    H_data_set, x_data_set = np.histogram(samples_test, bins=bins)
    H_generated_samples, _ = np.histogram(samples1, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var