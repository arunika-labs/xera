

from __future__ import annotations
import jax
import jax.numpy as jnp
from ..core import Module, param
from .. import initializers


def _dimension_numbers(ndim: int) -> jax.lax.ConvDimensionNumbers:
    """Rank-agnostic, channel-last dimension numbers for
    `jax.lax.conv_general_dilated`, following the same construction Flax
    uses for its generic-rank Conv: for an `ndim`-dimensional array
    `(batch, *spatial, channels)`, batch is dim 0, channels is the last
    dim, and everything in between is spatial -- for both the input/output
    (`lhs`/`out`, layout `N...C`) and the kernel (`rhs`, layout
    `*spatial, in, out`, i.e. "HWIO" in the 2D case).
    """
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _as_tuple(v, n):
    return v if isinstance(v, tuple) else (v,) * n


class Conv(Module):
    """N-dimensional convolution, N inferred from `len(kernel_size)` --
    `kernel_size=(3,)` is a 1D conv, `(3, 3)` is 2D, `(3, 3, 3)` is 3D,
    etc. One class instead of Conv1d/Conv2d/Conv3d, since the rank is just
    a shape-dispatch detail of the same underlying operation
    (`jax.lax.conv_general_dilated` is rank-agnostic natively) -- same
    reasoning as why MuonCore is one class handling ndim 2/3/4, not three
    separate optimizer classes.

    Input/output layout is channel-last: `(batch, *spatial, channels)`,
    matching `Dense`'s convention that the last axis is the feature
    dimension. Kernel (weight) shape is `(*kernel_size, in_channels //
    groups, out_channels)`.

    `padding` accepts `"SAME"`/`"VALID"` (passed straight through to
    `lax.conv_general_dilated`) or an explicit tuple of `(lo, hi)` pairs,
    one per spatial dimension.

    `groups` enables depthwise/grouped convolution (`in_channels % groups
    == 0`, `out_channels % groups == 0`) via `feature_group_count` --
    e.g. a depthwise conv is `groups=in_channels`.

    Transposed convolution isn't included here -- a `ConvTranspose` using
    `jax.lax.conv_transpose` would follow the same rank-agnostic pattern
    but needs a distinct code path, so it's left for a future addition
    rather than folded into this class.
    """

    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: int | tuple = 1
    padding: str | tuple = "SAME"
    dilation: int | tuple = 1
    groups: int = 1
    use_bias: bool = True

    def setup(self):
        assert self.in_channels % self.groups == 0, (
            f"in_channels ({self.in_channels}) must be divisible by groups ({self.groups})"
        )
        assert self.out_channels % self.groups == 0, (
            f"out_channels ({self.out_channels}) must be divisible by groups ({self.groups})"
        )

        weight_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
        self.weight = param(self.rng(), initializers.kaiming_normal(), weight_shape)
        self.bias = (
            param(self.rng(), initializers.zeros(), (self.out_channels,))
            if self.use_bias else None
        )

    def __call__(self, x):
        # x: (batch, *spatial, in_channels)
        n_spatial = len(self.kernel_size)
        stride = _as_tuple(self.stride, n_spatial)
        dilation = _as_tuple(self.dilation, n_spatial)
        padding = self.padding if isinstance(self.padding, str) else tuple(self.padding)

        y = jax.lax.conv_general_dilated(
            x, self.weight,
            window_strides=stride,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=_dimension_numbers(x.ndim),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


__all__ = ["Conv"]
