from ts_eval.viz.eval_table import EvalTable

from . import BaseComponent


class PredictionRankingsComponent(BaseComponent):
    component_type = "metrics"

    def __init__(
        self, points, time_slice, metric, metric_res, names, number_format="%#6.3g"
    ):
        self.points = points
        self.time_slice = time_slice
        self.metric = metric
        self.metric_res = metric_res
        self.names = names
        self.number_format = number_format

    def display(self):

        m_overall, m_steps = self.metric_res.overall(), self.metric_res.steps()
        eq_overall, eq_steps = self.metric_res.prediction_rankings()
        ref_overall, ref_steps = self.metric_res.ref_equality()

        data = []

        row = []
        for c in range(m_steps.shape[1]):
            value = self.number_format % m_overall[c]
            rank = eq_overall.ranks[c]
            # cast to get rid of numpy
            equality = bool(eq_overall.equality_bool_mask[c])
            ref_same = ref_overall[c]
            row += [self._format_row(value, rank, equality, ref_same, is_header=True)]

        data.append(row)

        for h in self.points:
            row = []
            for c in range(m_steps.shape[1]):
                value = self.number_format % m_steps[h, c]
                rank = eq_steps.ranks[h, c]
                # cast to get rid of numpy
                equality = bool(eq_steps.equality_bool_mask[h, c])
                ref_same = ref_steps[h, c]

                row += [
                    self._format_row(value, rank, equality, ref_same, is_header=False)
                ]

            data.append(row)

        t = EvalTable(
            data=data,
            headers=[str(n) for n in self.names],
            stubs=["Overall"] + list(self.points),
            title=f"{self.metric.name} on slice '{self.time_slice.name}'",
        )

        return t.as_html()

    def _format_row(self, value, rank, equality, ref_same, is_header):
        color = "green" if rank == 1 and equality is False else None
        color = "darkorange" if equality is True else color
        warn_sign = ref_same

        return (value, color, warn_sign, is_header, None)
