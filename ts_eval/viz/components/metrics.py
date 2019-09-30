from collections import OrderedDict

from statsmodels.iolib.table import SimpleTable

from . import BaseComponent


class MetricsComponent(BaseComponent):
    component_type = "metrics"

    def __init__(
        self,
        target,
        pred,
        points,
        time_slice,
        metrics,
        reference_pred,
        name,
        number_format="%#6.3g",
    ):
        self.target = target
        self.pred = pred
        self.points = points
        self.time_slice = time_slice
        self.metrics = metrics
        self.reference_pred = reference_pred
        self.name = name
        self.number_format = number_format

    def compute(self):
        self.point_metric_dict = OrderedDict()

        target = self.time_slice(self.target)
        reference_pred = self.time_slice(self.reference_pred)
        pred = self.time_slice(self.pred)

        # in theory, this could be vectorized, but it would make the whole thing too complex for nothing
        for metric in self.metrics:
            # TODO: assert reference_pred is not none... depending on the metric

            metric_res = metric(target, reference_pred, pred, points=self.points)
            self.point_metric_dict[metric.name] = metric_res

        return self.point_metric_dict

    def display(self):

        # Reformat the data in the following way:
        # - transpose
        # - convert numbers to strings, for compatibility reasons (SimpleTable)
        points_tuples = (
            list(zip(v.steps, v.steps_is_same)) for v in self.point_metric_dict.values()
        )
        p_metrics_data = list(map(self._format_list_of_numbers, zip(*points_tuples)))

        t = SimpleTable(
            data=p_metrics_data,
            headers=self.point_metric_dict.keys(),
            stubs=self.points,
            title=f"Point Metrics – {self.name} – {self.time_slice.name}",
        )

        # a hack to insert another header (could be passed initially, alongside datatypes param, but it fails for me)
        points_tuples = (
            (v.overall, v.overall_is_same) for v in self.point_metric_dict.values()
        )
        overall_data = self._format_list_of_numbers(points_tuples)
        t.insert(1, ["Overall"] + overall_data, datatype="header")

        return t.as_html()

    def _format_list_of_numbers(self, num_lst):
        return [
            self.number_format % el[0] + self._format_asterisk(el[1]) for el in num_lst
        ]

    def _format_asterisk(self, val):
        return "*" if val else "&nbsp;"
