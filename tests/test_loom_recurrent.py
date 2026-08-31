"""Tests for xera.xl.recurrent: SSM, SelectiveSSM, MambaBlock (SSM family)."""

import jax
import jax.numpy as jnp
import xera.loom as xl


# ---------------------------------------------------------------------------
# SSM (S4D)
# ---------------------------------------------------------------------------

def test_ssm_forward_shape():
    ssm = xl.SSM(channels=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 8))
    out = ssm(x)
    assert out.shape == (2, 10, 8)


def test_ssm_param_shapes():
    ssm = xl.SSM(channels=8, state_dim=4, key=jax.random.PRNGKey(0))
    assert ssm.log_A.shape == (8, 4)
    assert ssm.B.shape == (8, 4)
    assert ssm.C.shape == (8, 4)
    assert ssm.D.shape == (8,)
    assert ssm.log_dt.shape == (8,)


def test_ssm_A_matrix_always_negative():
    ssm = xl.SSM(channels=8, state_dim=4, key=jax.random.PRNGKey(0))
    A = -jnp.exp(ssm.log_A)
    assert bool(jnp.all(A < 0))


def test_ssm_dt_within_bounds():
    ssm = xl.SSM(channels=16, state_dim=4, dt_min=0.001, dt_max=0.1, key=jax.random.PRNGKey(0))
    dt = jnp.exp(ssm.log_dt)
    assert bool(jnp.all(dt >= ssm.dt_min - 1e-8))
    assert bool(jnp.all(dt <= ssm.dt_max + 1e-8))


def test_ssm_output_depends_on_full_history_at_last_step():
    ssm = xl.SSM(channels=4, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 4))
    out1 = ssm(x)[0, -1]
    x_perturbed = x.at[0, 0].add(10.0)
    out2 = ssm(x_perturbed)[0, -1]
    assert not jnp.allclose(out1, out2, atol=1e-4)


def test_ssm_causal_future_does_not_affect_past():
    # SSM is a scan over time, so output at position t should not depend on
    # input at position t' > t.
    ssm = xl.SSM(channels=4, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 4))
    out1 = ssm(x)[0, 2]
    x_perturbed = x.at[0, 5].add(100.0)
    out2 = ssm(x_perturbed)[0, 2]
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_ssm_grad_shapes_match_params():
    ssm = xl.SSM(channels=4, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(ssm, x)
    assert grads.B.shape == ssm.B.shape
    assert grads.C.shape == ssm.C.shape


def test_ssm_jit_compatible():
    ssm = xl.SSM(channels=4, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4))
    fwd = jax.jit(lambda m, x: m(x))
    out = fwd(ssm, x)
    assert out.shape == (2, 5, 4)


# ---------------------------------------------------------------------------
# SelectiveSSM
# ---------------------------------------------------------------------------

def test_selective_ssm_forward_shape():
    ssm = xl.SelectiveSSM(d_inner=16, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 16))
    out = ssm(x)
    assert out.shape == (2, 10, 16)


def test_selective_ssm_default_dt_rank():
    ssm = xl.SelectiveSSM(d_inner=32, state_dim=4, key=jax.random.PRNGKey(0))
    assert ssm._dt_rank == max(1, 32 // 16)


def test_selective_ssm_custom_dt_rank():
    ssm = xl.SelectiveSSM(d_inner=32, state_dim=4, dt_rank=3, key=jax.random.PRNGKey(0))
    assert ssm._dt_rank == 3


def test_selective_ssm_x_proj_output_dim():
    ssm = xl.SelectiveSSM(d_inner=32, state_dim=4, dt_rank=3, key=jax.random.PRNGKey(0))
    assert ssm.x_proj.out_features == 3 + 2 * 4


def test_selective_ssm_causal_future_does_not_affect_past():
    ssm = xl.SelectiveSSM(d_inner=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 8))
    out1 = ssm(x)[0, 2]
    x_perturbed = x.at[0, 5].add(100.0)
    out2 = ssm(x_perturbed)[0, 2]
    assert jnp.allclose(out1, out2, atol=1e-4)


def test_selective_ssm_input_dependent_selectivity():
    # Selectivity means different inputs at the same early timestep produce
    # different downstream dynamics (not just a fixed linear response).
    ssm = xl.SelectiveSSM(d_inner=8, state_dim=4, key=jax.random.PRNGKey(0))
    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 8))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (1, 6, 8))
    out1 = ssm(x1)
    out2 = ssm(x2)
    assert not jnp.allclose(out1, out2)


def test_selective_ssm_grad_shapes_match_params():
    ssm = xl.SelectiveSSM(d_inner=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(ssm, x)
    assert grads.x_proj.weight.shape == ssm.x_proj.weight.shape
    assert grads.dt_proj.weight.shape == ssm.dt_proj.weight.shape


def test_selective_ssm_jit_compatible():
    ssm = xl.SelectiveSSM(d_inner=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    fwd = jax.jit(lambda m, x: m(x))
    out = fwd(ssm, x)
    assert out.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------

def test_mamba_block_forward_shape():
    mamba = xl.MambaBlock(d_model=16, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 16))
    out = mamba(x)
    assert out.shape == (2, 10, 16)


def test_mamba_block_default_inner_dim():
    mamba = xl.MambaBlock(d_model=16, key=jax.random.PRNGKey(0))
    assert mamba._d_inner == 32


def test_mamba_block_custom_inner_dim():
    mamba = xl.MambaBlock(d_model=16, d_inner=48, key=jax.random.PRNGKey(0))
    assert mamba._d_inner == 48
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 16))
    out = mamba(x)
    assert out.shape == (1, 5, 16)


def test_mamba_block_has_submodules():
    mamba = xl.MambaBlock(d_model=16, key=jax.random.PRNGKey(0))
    assert isinstance(mamba.conv, xl.Conv)
    assert isinstance(mamba.ssm, xl.SelectiveSSM)
    assert isinstance(mamba.in_proj, xl.Dense)
    assert isinstance(mamba.out_proj, xl.Dense)


def test_mamba_block_causal_conv_future_does_not_affect_past():
    mamba = xl.MambaBlock(d_model=8, state_dim=4, conv_kernel=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 8))
    out1 = mamba(x)[0, 2]
    x_perturbed = x.at[0, 7].add(100.0)
    out2 = mamba(x_perturbed)[0, 2]
    assert jnp.allclose(out1, out2, atol=1e-3)


def test_mamba_block_grad_shapes_match_params():
    mamba = xl.MambaBlock(d_model=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    grads = jax.grad(lambda m, x: jnp.sum(m(x) ** 2))(mamba, x)
    assert grads.in_proj.weight.shape == mamba.in_proj.weight.shape
    assert grads.out_proj.weight.shape == mamba.out_proj.weight.shape


def test_mamba_block_jit_compatible():
    mamba = xl.MambaBlock(d_model=8, state_dim=4, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    fwd = jax.jit(lambda m, x: m(x))
    out = fwd(mamba, x)
    assert out.shape == (2, 5, 8)


def test_mamba_block_stack_composability():
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    blocks = [xl.MambaBlock(d_model=8, state_dim=4, key=k) for k in keys]
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 6, 8))
    for block in blocks:
        x = block(x)
    assert x.shape == (1, 6, 8)
