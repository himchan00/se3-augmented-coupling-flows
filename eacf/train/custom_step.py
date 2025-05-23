from typing import Callable, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax

from eacf.utils.base import FullGraphSample
from eacf.train.base import get_tree_leaf_norm_info
from eacf.utils.optimize import CustomOptimizerState

Params = chex.ArrayTree


def training_step(
    params: Params,
    x: FullGraphSample,
    opt_state: optax.OptState,
    key: chex.PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[
        [chex.PRNGKey, chex.ArrayTree, FullGraphSample], Tuple[chex.Array, dict]
    ],
    verbose_info: Optional[bool] = False,
    use_pmap: bool = False,
    pmap_axis_name: str = 'data'
) -> Tuple[Params, optax.OptState, dict]:
    """Compute loss and gradients and update model parameters.

    Args:
        params (AugmentedFlowParams): _description_
        x (FullGraphSample): _description_
        opt_state (optax.OptState): _description_
        key (chex.PRNGKey): _description_
        optimizer (optax.GradientTransformation): _description_
        loss_fn
        verbose_info
        use_pmap: whether the training step function is pmapped, such that gradient aggregation is needed.
        pmap_axis_name: name of axis for gradient aggregation across devices.


    Returns:
        Tuple[AugmentedFlowParams, optax.OptState, dict]: _description_
    """

    grad, info = jax.grad(loss_fn, has_aux=True, argnums=1)(
        key, params, x, verbose_info
    )
    if use_pmap:
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(
        grad_norm=optax.global_norm(grad),
        update_norm=optax.global_norm(updates),
        param_norm=optax.global_norm(params),
    )

    if verbose_info:
        info.update(
            {
                "grad_" + key: value
                for key, value in get_tree_leaf_norm_info(grad).items()
            }
        )
        info.update(
            {
                "update_" + key: value
                for key, value in get_tree_leaf_norm_info(updates).items()
            }
        )
    if isinstance(opt_state, CustomOptimizerState):
        info.update(ignored_grad_count=opt_state.ignored_grads_count,
                    total_optimizer_steps=opt_state.total_steps)
    return new_params, new_opt_state, info


def mean_with_mask(array: chex.Array, mask: chex.Array):
    chex.assert_rank(mask, 1)
    chex.assert_axis_dimension(array, 0, mask.shape[0])

    mask = jnp.expand_dims(mask, [1 + i for i in range(array.ndim - 1)])
    broadcasted_mask = jnp.broadcast_to(mask, array.shape)

    array = jnp.where(broadcasted_mask, array, jnp.zeros_like(array))
    return jnp.sum(array, axis=0) / mask.shape[0]


def training_step_with_masking(
    params: Params,
    x: FullGraphSample,
    opt_state: optax.OptState,
    key: chex.PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn_with_mask: Callable[
        [chex.PRNGKey, chex.ArrayTree, FullGraphSample], Tuple[chex.Array, Tuple[chex.Array, dict]]
    ],
    verbose_info: Optional[bool] = False,
    use_pmap: bool = False,
    pmap_axis_name: str = 'data'
) -> Tuple[Params, optax.OptState, dict]:
    """Compute loss and gradients and update model parameters.

    Args:
        params (AugmentedFlowParams): _description_
        x (FullGraphSample): _description_
        opt_state (optax.OptState): _description_
        key (chex.PRNGKey): _description_
        optimizer (optax.GradientTransformation): _description_
        loss_fn_with_mask: returns loss per sample in batch with an additional mask.
        verbose_info
        use_pmap: whether the training step function is pmapped, such that gradient aggregation is needed.
        pmap_axis_name: name of axis for gradient aggregation across devices.


    Returns:
        Tuple[AugmentedFlowParams, optax.OptState, dict]: _description_
    """
    key_batch = jax.random.split(key, x.positions.shape[0])
    grads, (masks, infos) = jax.vmap(jax.grad(loss_fn_with_mask, has_aux=True, argnums=1),
                                     in_axes=(0, None, 0, None))(
        key_batch, params, x, verbose_info
    )
    grad = jax.tree_util.tree_map(lambda x: mean_with_mask(x, masks), grads)
    info = jax.tree_util.tree_map(lambda x: mean_with_mask(x, masks), infos)
    info.update(masked_points=jnp.sum(~masks))

    if use_pmap:
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(
        grad_norm=optax.global_norm(grad),
        update_norm=optax.global_norm(updates),
        param_norm=optax.global_norm(params),
    )

    if verbose_info:
        info.update(
            {
                "grad_" + key: value
                for key, value in get_tree_leaf_norm_info(grad).items()
            }
        )
        info.update(
            {
                "update_" + key: value
                for key, value in get_tree_leaf_norm_info(updates).items()
            }
        )
    if isinstance(opt_state, CustomOptimizerState):
        info.update(ignored_grad_count=opt_state.ignored_grads_count,
                    total_optimizer_steps=opt_state.total_steps)
    return new_params, new_opt_state, info
