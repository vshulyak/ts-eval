from functools import partial

from ts_eval.metrics import ae, is_, se

from .factories import absolute_metric, adaptor_interval_metric, relative_metric


def partial_with_name(fn, name, **kwargs):
    """
    Creates a partials with a 'name' property
    """
    p = partial(fn, **kwargs)
    p.name = name
    return p


METRICS = [
    partial_with_name(absolute_metric, "MSE", metric_fn=se),
    partial_with_name(absolute_metric, "MAE", metric_fn=ae),
    partial_with_name(
        absolute_metric, "MIS", metric_fn=is_, adaptor=adaptor_interval_metric
    ),
    partial_with_name(relative_metric, "rMSE", metric_fn=se),
    partial_with_name(relative_metric, "rMAE", metric_fn=ae),
    partial_with_name(relative_metric, "FVrMSE", metric_fn=se, fv=True),
    partial_with_name(relative_metric, "FVrMAE", metric_fn=ae, fv=True),
    partial_with_name(
        relative_metric,
        "FVrMIS",
        metric_fn=is_,
        fv=True,
        adaptor=adaptor_interval_metric,
    ),
]

for metric in METRICS:
    globals()[metric.name] = metric


__all__ = [metric.name for metric in METRICS]
