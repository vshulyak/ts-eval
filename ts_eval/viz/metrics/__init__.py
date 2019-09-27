from functools import partial

from .defs import ae, rMIS, se
from .factories import absolute_metric, relative_metric

MSE = partial(absolute_metric, metric_fn=se)
MAE = partial(absolute_metric, metric_fn=ae)
rMSE = partial(relative_metric, metric_fn=se)
rMAE = partial(relative_metric, metric_fn=ae)
FVrMSE = partial(relative_metric, metric_fn=se, fv=True)
FVrMAE = partial(relative_metric, metric_fn=ae, fv=True)
FVrMIS = partial(rMIS, fv=True)
