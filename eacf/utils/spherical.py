from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from eacf.utils.numerical import safe_norm
from eacf.utils.numerical import rotate_2d

def to_spherical_and_log_det(x: chex.Array, reference: chex.Array,
                             parity_invariant: bool = False) -> Tuple[chex.Array, chex.Array]:
    chex.assert_rank(x, 1)
    chex.assert_rank(reference, 2)
    dim = x.shape[0]
    if dim == 3:
        return _to_spherical_and_log_det(x, reference, parity_invariant)
    else:
        assert dim == 2
        return _to_polar_and_log_det(x, reference)



def to_cartesian_and_log_det(sph_x: chex.Array, reference: chex.Array, parity_invariant: bool = False) -> \
        Tuple[chex.Array, chex.Array]:
    chex.assert_rank(sph_x, 1)
    chex.assert_rank(reference, 2)
    dim = sph_x.shape[0]
    if dim == 3:
        return _to_cartesian_and_log_det(sph_x, reference, parity_invariant)
    else:
        assert dim == 2
        return polar_to_cartesian_and_log_det(sph_x, reference)



def _to_polar_and_log_det(x: chex.Array,
                          reference: chex.Array,
                          ) -> Tuple[chex.Array, chex.Array]:
    chex.assert_shape(x, (2,))
    origin, y = jnp.split(reference, (1,), axis=-2)
    y, origin = jnp.squeeze(y), jnp.squeeze(origin)
    chex.assert_equal_shape((origin, y, x))

    vector_x = x - origin
    vector_y = y - origin

    # Calculate radius.
    r = safe_norm(vector_x, axis=-1)
    unit_vector_x = vector_x / r


    # Calculate angle
    norm_y = safe_norm(y, axis=-1)
    unit_vector_y_axis = vector_y / norm_y
    x_proj_norm = jnp.dot(unit_vector_x, unit_vector_y_axis)
    # Norm in direction perpendicular to x.
    perp_line = jnp.cross(unit_vector_y_axis, unit_vector_x)

    theta = jnp.arctan2(perp_line, x_proj_norm)
    log_det = - jnp.log(r)


    x_polar = jnp.stack([r, theta])
    return x_polar, log_det


def polar_to_cartesian_and_log_det(
        x_polar: chex.Array,
        reference: chex.Array) -> Tuple[chex.Array, chex.Array]:
    chex.assert_shape(x_polar, (2,))
    origin, y = jnp.split(reference, (1,), axis=-2)
    y, origin = jnp.squeeze(y), jnp.squeeze(origin)
    chex.assert_equal_shape((origin, y, x_polar))
    y_vector = y - origin

    r, theta = x_polar
    y_unit_vec = y_vector / safe_norm(y_vector)

    log_det = jnp.log(r)
    x_vector = rotate_2d(r * y_unit_vec, theta)
    x = origin + x_vector
    return x, log_det


def _to_spherical_and_log_det(
        x: chex.Array,
        reference: chex.Array,
        enforce_parity_invariance: bool = False,
                              ) -> Tuple[chex.Array, chex.Array]:
    """Note that if `enforce_parity_invariance` is True we use z - (0, 0, 0) to obtain another vector.
    This only works if we assume that (0, 0, 0) is our centre of mass (i.e. this will only work within a flow layer
    that ensures that z - (0,0,0) is an equivariant quantity)."""

    chex.assert_rank(x, 1)
    dim = x.shape[0]
    origin, z, o = jnp.split(reference, (1,2), axis=-2)
    origin, z, o = jnp.squeeze(origin, axis=-2), jnp.squeeze(z, axis=-2), jnp.squeeze(o, axis=-2)
    chex.assert_equal_shape([x, origin, z, o])

    # Get z, y, and x axes (unit vectors).
    z_vector = z - origin
    z_axis_vector = z_vector / safe_norm(z_vector)
    o_vector = o - origin
    x_vector = o_vector - z_axis_vector * jnp.dot(o_vector, z_axis_vector)  # vector rejection.
    x_axis_vector = x_vector / safe_norm(x_vector)
    y_vector = jnp.cross(x_axis_vector, z_axis_vector)
    y_axis_vector = y_vector / safe_norm(y_vector)
    if enforce_parity_invariance:
        # The cross product returns a pseudo-vector. Multiplying this by
        # A pseudo-scalar then converts this back into a normal (polar) vector.
        # To get the pseudo-scalar we take the sign of the dot product between the pseudo-vector
        # and a polar vector. The polar vector can be anything, as long as it is not orthogonal to
        # the pseudo-vector. We use the vector from the z reference point to centre (0,0,0) as
        # the polar vector (we can't use `x_axis_vector` or `z_axis_vector` as these are orthogonal to
        # `y_axis_vector`).
        pseudo_scalar = jnp.sign(jnp.dot(y_axis_vector, z))
        y_axis_vector = y_axis_vector * pseudo_scalar


    vector = x - origin
    r = safe_norm(vector)
    vector_in_z_dir = z_axis_vector * jnp.dot(vector, z_axis_vector)
    vector_z_perp = vector - vector_in_z_dir  # vector rejection
    theta = jnp.arctan2(safe_norm(vector_z_perp), jnp.dot(vector_in_z_dir, z_axis_vector))

    a = jnp.dot(vector_z_perp, y_axis_vector)  # magnitude of vector along y axis.
    b = jnp.dot(vector_z_perp, x_axis_vector)  # magnitude of vector along x axis
    torsion = jnp.arctan2(a, b)

    x = jnp.stack([r, theta, torsion])
    log_det = - (2*jnp.log(r) + jnp.log(jnp.sin(theta)))
    return x, jnp.squeeze(log_det)


def _to_cartesian_and_log_det(sph_x: chex.Array, reference: chex.Array,
                              enforce_parity_invariance: bool = False) -> \
        Tuple[chex.Array, chex.Array]:
    """Note that if `enforce_parity_invariance` is True we use z - (0, 0, 0) to obtain another vector.
    This only works if we assume that (0, 0, 0) is our centre of mass (i.e. this will only work within a flow layer
    that ensures that z - (0,0,0) is an equivariant quantity)."""

    chex.assert_rank(sph_x, 1)
    origin, z, o = jnp.split(reference, (1, 2), axis=-2)
    origin, z, o = jnp.squeeze(origin, axis=-2), jnp.squeeze(z, axis=-2), jnp.squeeze(o, axis=-2)
    chex.assert_equal_shape([sph_x, origin, z, o])

    # Get z, y, and x axes (unit vectors).
    z_vector = z - origin
    z_axis_vector = z_vector / safe_norm(z_vector)
    o_vector = o - origin
    x_vector = o_vector - z_axis_vector * jnp.dot(o_vector, z_axis_vector)  # vector rejection.
    x_axis_vector = x_vector / safe_norm(x_vector)
    y_vector = jnp.cross(x_axis_vector, z_axis_vector)
    y_axis_vector = y_vector / safe_norm(y_vector)
    if enforce_parity_invariance:
        # The cross product returns a pseudo-vector. Multiplying this by
        # A pseudo-scalar then converts this back into a normal (polar) vector.
        # To get the pseudo-scalar we take the sign of the dot product between the pseudo-vector
        # and a polar vector. The polar vector can be anything, as long as it is not orthogonal to
        # the pseudo-vector. We use the vector from the z reference point to centre (0,0,0) as
        # the polar vector (we can't use `x_axis_vector` or `z_axis_vector` as these are orthogonal to
        # `y_axis_vector`).
        pseudo_scalar = jnp.sign(jnp.dot(y_axis_vector, z))
        y_axis_vector = y_axis_vector * pseudo_scalar

    r, theta, torsion = jnp.split(sph_x, 3)
    r, theta, torsion = jax.tree_util.tree_map(jnp.squeeze, (r, theta, torsion))

    vector_z_perp = x_axis_vector * jnp.cos(torsion) + y_axis_vector * jnp.sin(torsion)  # Should have norm of 1.
    vector = r*(z_axis_vector * jnp.cos(theta) + vector_z_perp * jnp.sin(theta))

    x_cartesian = vector + origin

    log_det = (2*jnp.log(r) + jnp.log(jnp.sin(theta)))
    return x_cartesian, jnp.squeeze(log_det)
