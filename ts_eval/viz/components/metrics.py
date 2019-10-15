from ts_eval.viz.eval_table import EvalTable

from . import BaseComponent


class MetricsComponent(BaseComponent):
    component_type = "metrics"

    def __init__(
        self,
        points,
        time_slice,
        metrics,
        name,
        metric_res,
        pred_idx,
        number_format="%#6.3g",
    ):
        self.points = points
        self.time_slice = time_slice
        self.metrics = metrics
        self.name = name
        self.metric_res = metric_res
        self.pred_idx = pred_idx
        self.number_format = number_format

    def display(self):

        data = []

        row = []
        for v in self.metric_res.values():
            m_overall = v.overall()
            # TODO: cache
            eq_overall, _ = v.prediction_rankings()
            ref_overall, _ = v.ref_equality()

            value = self.number_format % m_overall[self.pred_idx]
            rank = eq_overall.ranks[self.pred_idx]
            # cast to get rid of numpy
            equality = bool(eq_overall.equality_bool_mask[self.pred_idx])
            ref_same = ref_overall[self.pred_idx]
            row += [self._format_row(value, rank, equality, ref_same, is_header=True)]

        data.append(row)

        for h in self.points:
            row = []
            for v in self.metric_res.values():
                m_steps = v.steps()
                # TODO: cache
                _, eq_steps = v.prediction_rankings()
                _, ref_steps = v.ref_equality()

                value = self.number_format % m_steps[h, self.pred_idx]
                rank = eq_steps.ranks[h, self.pred_idx]
                # cast to get rid of numpy
                equality = bool(eq_steps.equality_bool_mask[h, self.pred_idx])
                ref_same = ref_steps[h, self.pred_idx]

                row += [
                    self._format_row(value, rank, equality, ref_same, is_header=False)
                ]

            data.append(row)

        t = EvalTable(
            data=data,
            headers=self.metric_res.keys(),
            stubs=["Overall"] + list(self.points),
            title=f"{self.name} on slice '{self.time_slice.name}'",
        )

        return t.as_html()

    def _format_row(self, value, rank, equality, ref_same, is_header):
        color = "green" if rank == 1 and equality is False else None
        color = "darkorange" if equality is True else color
        warn_sign = ref_same

        return (value, color, warn_sign, is_header, None)
