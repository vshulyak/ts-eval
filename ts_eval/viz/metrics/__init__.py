from functools import partial

from .defs import ae, rMIS, se
from .factories import absolute_metric, relative_metric


def partial_with_name(fn, name, **kwargs):
    """
    Creates a partials with a 'name' property
    """
    p = partial(fn, **kwargs)
    p.name = name
    return p


MSE = partial_with_name(absolute_metric, "MSE", metric_fn=se)
MAE = partial_with_name(absolute_metric, "MAE", metric_fn=ae)
rMSE = partial_with_name(relative_metric, "rMSE", metric_fn=se)
rMAE = partial_with_name(relative_metric, "rMAE", metric_fn=ae)
FVrMSE = partial_with_name(relative_metric, "FVrMSE", metric_fn=se, fv=True)
FVrMAE = partial_with_name(relative_metric, "FVrMAE", metric_fn=ae, fv=True)
FVrMIS = partial_with_name(rMIS, "FVrMIS", fv=True)
