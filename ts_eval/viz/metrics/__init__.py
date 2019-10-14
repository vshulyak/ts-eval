from ts_eval.metrics import ae, is_, se

from .adaptors import adaptor_interval_metric, adaptor_point_metric


def metric_factory(fn, name, relative=False, fv=False, adaptor=adaptor_point_metric):
    def fnw(*args, **kwargs):
        return fn(*args, **kwargs)

    fnw.__doc__ = fn.__doc__
    fnw.name = name
    fnw.relative = relative
    fnw.fv = fv
    fnw.adaptor = adaptor
    return fnw


METRICS = [
    metric_factory(se, "MSE"),
    metric_factory(ae, "MAE"),
    metric_factory(se, "rMSE", relative=True),
    metric_factory(ae, "rMAE", relative=True),
    metric_factory(se, "FVrMSE", relative=True, fv=True),
    metric_factory(ae, "FVrMAE", relative=True, fv=True),
    metric_factory(is_, "MIS", adaptor=adaptor_interval_metric),
    metric_factory(is_, "rMIS", relative=True, adaptor=adaptor_interval_metric),
    metric_factory(
        is_, "FVrMIS", relative=True, fv=True, adaptor=adaptor_interval_metric
    ),
]


for metric in METRICS:
    globals()[metric.name] = metric


__all__ = [metric.name for metric in METRICS]
