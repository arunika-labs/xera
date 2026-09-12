"""Tests for xera.functional.loss: flat *_loss functions."""

import jax
import jax.numpy as jnp
import xera.functional as F


def test_l1_matches_manual_mae():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.5, 2.0])
    expected = jnp.mean(jnp.abs(pred - target))
    assert jnp.allclose(F.mae_loss(pred, target), expected)


def test_l1_zero_for_identical_inputs():
    x = jnp.array([1.0, -2.0, 3.5])
    assert jnp.allclose(F.mae_loss(x, x), 0.0)


def test_l2_matches_manual_mse():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.5, 2.0])
    expected = jnp.mean(jnp.square(pred - target))
    assert jnp.allclose(F.mse_loss(pred, target), expected)


def test_rmse_is_sqrt_of_l2():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([0.0, 2.0, 5.0])
    assert jnp.allclose(F.rmse_loss(pred, target), jnp.sqrt(F.mse_loss(pred, target)))


def test_ce_with_integer_labels():
    logits = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = jnp.array([0, 1])
    loss = F.ce_loss(logits, labels)
    assert loss.shape == ()
    assert float(loss) > 0.0


def test_ce_with_onehot_labels_matches_integer_labels():
    logits = jnp.array([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]])
    labels_int = jnp.array([0, 1])
    labels_onehot = jax.nn.one_hot(labels_int, 3)
    assert jnp.allclose(F.ce_loss(logits, labels_int), F.ce_loss(logits, labels_onehot))


def test_ce_lower_for_confident_correct_predictions():
    labels = jnp.array([0, 1])
    confident_logits = jnp.array([[10.0, 0.0], [0.0, 10.0]])
    unsure_logits = jnp.array([[0.1, 0.0], [0.0, 0.1]])
    assert float(F.ce_loss(confident_logits, labels)) < float(F.ce_loss(unsure_logits, labels))


def test_bce_matches_bce_with_logits():
    logits = jnp.array([0.5, -1.0, 2.0])
    labels = jnp.array([1.0, 0.0, 1.0])
    assert jnp.allclose(F.bce_loss(logits, labels), F.bce_with_logits_loss(logits, labels))


def test_bce_lower_for_correct_confident_predictions():
    labels = jnp.array([1.0, 0.0])
    confident_logits = jnp.array([10.0, -10.0])
    unsure_logits = jnp.array([0.1, -0.1])
    assert float(F.bce_loss(confident_logits, labels)) < float(F.bce_loss(unsure_logits, labels))


def test_hinge_zero_when_margin_satisfied():
    pred = jnp.array([2.0, -2.0])
    target = jnp.array([1.0, -1.0])
    assert jnp.allclose(F.hinge_loss(pred, target, margin=1.0), 0.0)


def test_hinge_positive_when_margin_violated():
    pred = jnp.array([0.1])
    target = jnp.array([1.0])
    loss = F.hinge_loss(pred, target, margin=1.0)
    assert float(loss) > 0.0


def test_huber_matches_l2_for_small_errors():
    pred = jnp.array([1.0])
    target = jnp.array([1.1])
    huber = F.huber_loss(pred, target, delta=1.0)
    l2_half = 0.5 * jnp.square(pred - target).mean()
    assert jnp.allclose(huber, l2_half, atol=1e-5)


def test_huber_linear_for_large_errors():
    pred = jnp.array([0.0])
    target = jnp.array([100.0])
    delta = 1.0
    huber = F.huber_loss(pred, target, delta=delta)
    expected = 0.5 * delta ** 2 + delta * (100.0 - delta)
    assert jnp.allclose(huber, expected, atol=1e-3)


def test_smooth_l1_small_errors_quadratic():
    pred = jnp.array([1.0])
    target = jnp.array([1.2])
    beta = 1.0
    loss = F.smooth_l1_loss(pred, target, beta=beta)
    expected = 0.5 * (0.2 ** 2) / beta
    assert jnp.allclose(loss, expected, atol=1e-5)


def test_smooth_l1_large_errors_linear():
    pred = jnp.array([0.0])
    target = jnp.array([10.0])
    beta = 1.0
    loss = F.smooth_l1_loss(pred, target, beta=beta)
    expected = 10.0 - 0.5 * beta
    assert jnp.allclose(loss, expected, atol=1e-5)


def test_kldiv_zero_for_identical_distributions():
    probs = jnp.array([[0.2, 0.3, 0.5]])
    log_probs = jnp.log(probs)
    loss = F.kl_div_loss(log_probs, probs)
    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_kldiv_positive_for_different_distributions():
    log_probs = jnp.log(jnp.array([[0.9, 0.1]]))
    target = jnp.array([[0.1, 0.9]])
    loss = F.kl_div_loss(log_probs, target)
    assert float(loss) > 0.0


def test_nll_with_integer_labels_positive():
    log_probs = jax.nn.log_softmax(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
    labels = jnp.array([0, 1])
    loss = F.nll_loss(log_probs, labels)
    assert float(loss) > 0.0


def test_nll_matches_ce_via_log_softmax():
    logits = jnp.array([[1.0, 0.5, -0.5], [0.2, -0.3, 1.1]])
    labels = jnp.array([0, 2])
    log_probs = jax.nn.log_softmax(logits)
    assert jnp.allclose(F.nll_loss(log_probs, labels), F.ce_loss(logits, labels))


def test_focal_loss_downweights_easy_examples_relative_to_ce():
    logits = jnp.array([[10.0, 0.0]])  # very confident, correct
    labels = jnp.array([0])
    focal = F.focal_loss(logits, labels, alpha=0.25, gamma=2.0)
    ce = F.ce_loss(logits, labels)
    assert float(focal) < float(ce)


def test_focal_loss_gamma_zero_reduces_toward_weighted_ce():
    logits = jnp.array([[1.0, -1.0]])
    labels = jnp.array([0])
    focal = F.focal_loss(logits, labels, alpha=0.5, gamma=0.0)
    assert float(focal) > 0.0


def test_cosine_embedding_zero_for_identical_similar_vectors():
    v = jnp.array([[1.0, 0.0, 0.0]])
    target = jnp.array([1])
    loss = F.cosine_embedding_loss(v, v, target)
    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_cosine_embedding_dissimilar_pair_below_margin_is_zero():
    v1 = jnp.array([[1.0, 0.0]])
    v2 = jnp.array([[-1.0, 0.0]])  # cosine = -1, well below margin=0
    target = jnp.array([-1])
    loss = F.cosine_embedding_loss(v1, v2, target, margin=0.0)
    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_margin_ranking_zero_when_correctly_ranked_beyond_margin():
    pred1 = jnp.array([3.0])
    pred2 = jnp.array([1.0])
    target = jnp.array([1.0])
    loss = F.margin_ranking_loss(pred1, pred2, target, margin=1.0)
    assert jnp.allclose(loss, 0.0)


def test_margin_ranking_positive_when_misranked():
    pred1 = jnp.array([1.0])
    pred2 = jnp.array([3.0])
    target = jnp.array([1.0])  # pred1 should rank higher but doesn't
    loss = F.margin_ranking_loss(pred1, pred2, target, margin=1.0)
    assert float(loss) > 0.0


def test_poisson_loss_runs_and_is_finite():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.0, 2.0, 4.0])
    loss = F.poisson_loss(pred, target)
    assert jnp.isfinite(loss)


def test_gamma_loss_runs_and_is_finite():
    pred = jnp.array([1.0, 2.0])
    target = jnp.array([1.5, 1.0])
    loss = F.gamma_loss(pred, target)
    assert jnp.isfinite(loss)


def test_logcosh_close_to_zero_for_small_errors():
    pred = jnp.array([1.0])
    target = jnp.array([1.001])
    loss = F.log_cosh_loss(pred, target)
    assert float(loss) < 1e-4


def test_logcosh_grows_for_large_errors():
    pred = jnp.array([0.0])
    target = jnp.array([10.0])
    loss = F.log_cosh_loss(pred, target)
    assert float(loss) > 1.0


def test_quantile_median_matches_half_l1():
    pred = jnp.array([2.0, 5.0])
    target = jnp.array([3.0, 4.0])
    loss = F.quantile_loss(pred, target, quantile=0.5)
    expected = 0.5 * jnp.mean(jnp.abs(pred - target))
    assert jnp.allclose(loss, expected)


def test_quantile_asymmetric_penalty_depends_on_quantile():
    # error = pred - target; for quantile=0.9, overshooting (pred > target)
    # is penalized more heavily than undershooting by the same margin.
    pred_under = jnp.array([1.0])
    pred_over = jnp.array([3.0])
    target = jnp.array([2.0])
    loss_under = F.quantile_loss(pred_under, target, quantile=0.9)
    loss_over = F.quantile_loss(pred_over, target, quantile=0.9)
    assert float(loss_over) > float(loss_under)


def test_sigmoid_focal_ce_lower_for_confident_correct():
    logits = jnp.array([10.0])
    labels = jnp.array([1.0])
    confident_loss = F.sigmoid_focal_cross_entropy_loss(logits, labels)
    logits_unsure = jnp.array([0.1])
    unsure_loss = F.sigmoid_focal_cross_entropy_loss(logits_unsure, labels)
    assert float(confident_loss) < float(unsure_loss)


def test_triplet_loss_zero_when_well_separated():
    anchor = jnp.array([[0.0, 0.0]])
    positive = jnp.array([[0.1, 0.0]])
    negative = jnp.array([[10.0, 10.0]])
    loss = F.triplet_loss(anchor, positive, negative, margin=1.0)
    assert jnp.allclose(loss, 0.0)


def test_triplet_loss_positive_when_negative_is_close():
    anchor = jnp.array([[0.0, 0.0]])
    positive = jnp.array([[5.0, 5.0]])
    negative = jnp.array([[0.1, 0.0]])
    loss = F.triplet_loss(anchor, positive, negative, margin=1.0)
    assert float(loss) > 0.0


def test_contrastive_loss_zero_for_similar_close_pair():
    pred1 = jnp.array([[0.0, 0.0]])
    pred2 = jnp.array([[0.0, 0.0]])
    target = jnp.array([1.0])
    loss = F.contrastive_loss(pred1, pred2, target, margin=1.0)
    assert jnp.allclose(loss, 0.0)


def test_contrastive_loss_zero_for_dissimilar_far_pair():
    pred1 = jnp.array([[0.0, 0.0]])
    pred2 = jnp.array([[10.0, 10.0]])
    target = jnp.array([0.0])
    loss = F.contrastive_loss(pred1, pred2, target, margin=1.0)
    assert jnp.allclose(loss, 0.0)


def test_contrastive_loss_positive_for_dissimilar_close_pair():
    pred1 = jnp.array([[0.0, 0.0]])
    pred2 = jnp.array([[0.1, 0.0]])
    target = jnp.array([0.0])
    loss = F.contrastive_loss(pred1, pred2, target, margin=1.0)
    assert float(loss) > 0.0


def test_loss_is_differentiable_via_grad():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 1.5, 3.5])
    grad_fn = jax.grad(lambda p: F.mse_loss(p, target))
    grads = grad_fn(pred)
    assert grads.shape == pred.shape
    assert not jnp.allclose(grads, jnp.zeros_like(grads))
