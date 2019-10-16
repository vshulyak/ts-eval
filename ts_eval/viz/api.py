import inspect

from collections import OrderedDict

from ts_eval.viz.metrics.metric_container import MetricContainer
from ts_eval.viz.utils import get_pretty_var_names

from . import time_slices as default_time_slices
from .components.dataset_description import DatasetDescriptionComponent
from .components.legend import LegendComponent
from .components.metrics import MetricsComponent
from .components.prediction_plot import PredictionPlotComponent
from .components.prediction_rankings import PredictionRankingsComponent
from .data_containers import xr_2d_factory, xr_3d_factory
from .layouts.jupyter_html import JupyterHTMLLayout


class TSMetrics(object):
    """
    Builder interface for merics components
    """

    def __init__(self, target, *preds, layout_class=JupyterHTMLLayout):
        self._target = target
        self._preds = preds
        self._ref = None
        self._ref_name = None
        self._layout_class = layout_class

        # default points of interest: first and last. Motivation: horizons can be quite big, we don't want to
        # draw too many graphs without explicit permission
        self.h = self._target.sizes["h"]
        self._points = [0, self.h - 1] if self.h > 2 else 0
        self._time_slices = [default_time_slices.all]
        self._components = OrderedDict()

        # get names
        self._names = get_pretty_var_names(
            target_vars=preds,
            local_vars=inspect.currentframe().f_back.f_locals.items(),
            fallback_name_prefix="Pred",
        )

    def _register_component(self, key, comp):
        if key in self._components:
            raise Exception(f"Component {key} already registered")
        self._components[key] = comp

    def use_reference(self, ref):
        """
        Sets the base model predictions for relative metrics (typically naive/snaive)
        """
        self._ref = ref
        self._ref_name = (
            get_pretty_var_names(
                target_vars=[ref],
                local_vars=inspect.currentframe().f_back.f_locals.items(),
                fallback_name_prefix="Ref",
            )[0]
            if self._ref_name is None
            else self._ref_name
        )
        return self

    def names(self, *names):
        """
        Sets verbose names to prediction datasets
        """
        assert len(names) == len(self._preds)
        self._names = names
        return self

    def ref_name(self, name):
        """
        Sets verbose names to reference dataset
        """
        self._ref_name = name if self._ref_name is None else self._ref_name
        return self

    def for_horizons(self, *points):
        """
        Sets horizons to display
        """
        assert all(
            [p >= 0 and p < self.h for p in points]
        ), f"Points of interest should be within 0 and {self.h-1}"

        self._points = points
        return self

    def for_time_slices(self, *time_slices):
        """
        Sets time time_slices we want to get merics for. Example: all, weekends, weekdays, holidays, etc.
        """
        # TODO: what do I do if it's defined last, not first? Other methods will not pick it up
        self._time_slices = time_slices
        return self

    def with_description(self):
        self._register_component(
            "description",
            DatasetDescriptionComponent(
                target=self._target,
                preds=self._preds,
                points=self._points,
                time_slices=None,
                names=self._names,
                ref=self._ref,
                ref_name=self._ref_name,
            ),
        )

        return self

    def with_metrics(self, *metrics):
        # TODO: assert metrics exist or callable
        # TODO: assert point metrics available for this container
        cnt = MetricContainer.build(
            self._target, self._ref, self._preds, self._time_slices, metrics
        )

        for s in self._time_slices:
            for i, d in enumerate(self._preds):
                self._register_component(
                    f"metrics_{i}_{s.name}",
                    MetricsComponent(
                        points=self._points,
                        time_slice=s,
                        name=self._names[i],
                        metrics=metrics,
                        metric_res=cnt.metric_res[s],
                        pred_idx=i,
                    ),
                )
        # legend should override another legend if already registered
        self._register_component(
            "legend",
            LegendComponent(
                target=self._target,
                preds=self._preds,
                points=self._points,
                time_slices=self._time_slices,
                names=self._names,
            ),
        )

        return self

    def with_prediction_rankings(self, *metrics):

        cnt = MetricContainer.build(
            self._target, self._ref, self._preds, self._time_slices, metrics
        )

        for s in self._time_slices:
            for metric in metrics:
                self._register_component(
                    f"prediction_rankings_{metric.name}_{s.name}",
                    PredictionRankingsComponent(
                        points=self._points,
                        time_slice=s,
                        names=self._names,
                        metric=metric,
                        metric_res=cnt.metric_res[s][metric.name],
                    ),
                )
        # legend should override another legend if already registered
        self._register_component(
            "legend",
            LegendComponent(
                target=self._target,
                preds=self._preds,
                points=self._points,
                time_slices=self._time_slices,
                names=self._names,
            ),
        )

        return self

    def with_predictions_plot(self, figsize=(14, 7), cm_name="Dark2", pi=True):
        self._register_component(
            "prediction_plot",
            PredictionPlotComponent(
                target=self._target,
                preds=self._preds,
                points=self._points,
                time_slices=self._time_slices,
                names=self._names,
                figsize=figsize,
                cm_name=cm_name,
                pi=pi,
            ),
        )

        return self

    def show(self):
        self.compute()  # this could be more elegant...
        return self._layout_class(self._components)

    def compute_metrics(self, *metrics):
        return MetricContainer.build(
            self._target, self._ref, self._preds, self._time_slices, metrics
        )

    def compute(self):
        res = {}
        for k, c in self._components.items():
            c_result = c.compute()
            if c_result:
                res[k] = c_result

        return type("TSMetricsRes", (), res)


def ts_inspect_2d(target, *preds, start_date=None, freq=None):
    """
    Builds TSMertics for point predictions only, creating internal representation for it.
    """
    return TSMetrics(
        xr_2d_factory(target, start_date=start_date, freq=freq),
        *[xr_2d_factory(p, start_date=start_date, freq=freq) for p in preds],
    )


def ts_inspect_3d(target, *preds, start_date=None, freq=None):
    """
    Builds TSMertics for point predictions and prediction intervals, creating internal representation for it.
    """
    return TSMetrics(
        xr_2d_factory(target, start_date=start_date, freq=freq),
        *[xr_3d_factory(p, start_date=start_date, freq=freq) for p in preds],
    )
