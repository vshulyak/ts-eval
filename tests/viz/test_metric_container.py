import numpy as np

from ts_eval.viz import metrics as mtx
from ts_eval.viz import time_slices
from ts_eval.viz.metrics.metric_container import MetricContainer, MetricResult


def _compute_scores(metric, target_slice, data_slices):
    scores = []
    for pred_slice in data_slices:
        scores += [metric(target_slice.mean_.values, pred_slice.mean_.values)]
    return np.stack(scores, 2)


def _randomize_xarray_mean(arr):
    return arr + np.random.random(arr.mean_.shape)


def test_metric_container__steps_overall(xarray_2d__index_dt):

    metric = mtx.MSE
    data_slices = [_randomize_xarray_mean(xarray_2d__index_dt)]
    scores = _compute_scores(
        metric=metric, target_slice=xarray_2d__index_dt, data_slices=data_slices
    )

    mr = MetricResult(ref=None, rest=scores, relative=metric.relative, fv=metric.fv)

    assert mr.steps().shape == (xarray_2d__index_dt.sizes["h"], len(data_slices))
    assert np.allclose(mr.steps().mean(), mr.overall())


def test_metric_container__2_slices(xarray_2d__index_dt):
    metric = mtx.MSE
    data_slices = [
        _randomize_xarray_mean(xarray_2d__index_dt),
        _randomize_xarray_mean(xarray_2d__index_dt),
        _randomize_xarray_mean(xarray_2d__index_dt),
    ]
    scores = _compute_scores(
        metric=metric, target_slice=xarray_2d__index_dt, data_slices=data_slices
    )

    mr = MetricResult(ref=None, rest=scores, relative=metric.relative, fv=metric.fv)

    assert mr.steps().shape == (xarray_2d__index_dt.sizes["h"], len(data_slices))


def test_metric_container__ref_equality(xarray_2d__index_dt):

    metric = mtx.MSE
    data_slices = [_randomize_xarray_mean(xarray_2d__index_dt)]
    ref = _randomize_xarray_mean(xarray_2d__index_dt)
    scores = _compute_scores(
        metric=metric, target_slice=xarray_2d__index_dt, data_slices=data_slices
    )
    ref_score = metric(xarray_2d__index_dt.mean_.values, ref.mean_.values)

    mr = MetricResult(
        ref=ref_score, rest=scores, relative=metric.relative, fv=metric.fv
    )

    overall, steps = mr.ref_equality()
    assert steps.shape == (xarray_2d__index_dt.sizes["h"], len(data_slices))
    assert overall.shape == (len(data_slices),)
    assert not np.isnan(steps).any()
    assert not np.isnan(overall).any()


def test_metric_container__prediction_rankings(xarray_2d__index_dt):

    metric = mtx.FVrMSE

    data_slices = [_randomize_xarray_mean(xarray_2d__index_dt)]
    scores = _compute_scores(
        metric=metric, target_slice=xarray_2d__index_dt, data_slices=data_slices
    )

    mr = MetricResult(ref=None, rest=scores, relative=metric.relative, fv=metric.fv)

    overall, steps = mr.prediction_rankings()

    assert steps.ranks.shape == (xarray_2d__index_dt.sizes["h"], len(data_slices))
    assert steps.equality_bool_mask.shape == (
        xarray_2d__index_dt.sizes["h"],
        len(data_slices),
    )
    assert overall.ranks.shape == (len(data_slices),)
    assert overall.equality_bool_mask.shape == (len(data_slices),)
    assert not np.isnan(steps.ranks).any()
    assert not np.isnan(overall.equality_bool_mask).any()
    assert not np.isnan(steps.ranks).any()
    assert not np.isnan(overall.equality_bool_mask).any()


def test_metric_container_list(xarray_2d__index_dt):
    """

    """
    target = xarray_2d__index_dt

    # [0] – reference
    ref = xarray_2d__index_dt - 5
    preds = [xarray_2d__index_dt + 6, xarray_2d__index_dt + 20]

    slices = [time_slices.all, time_slices.weekend]

    metrics = [mtx.MSE, mtx.rMSE, mtx.FVrMSE]

    cnt = MetricContainer.build(target, ref, preds, slices, metrics)

    assert set(cnt.metric_res.keys()) == set(slices)
    for mc in cnt.metric_res.values():
        assert set(mc.keys()) == set([m.name for m in metrics])
