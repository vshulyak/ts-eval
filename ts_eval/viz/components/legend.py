from ts_eval.viz.eval_table import EvalTable

from . import BaseComponent


class LegendComponent(BaseComponent):
    component_type = "metrics"

    def __init__(self, target, preds, points, time_slices, names):
        self.target = target
        self.preds = preds
        self.points = points
        self.time_slices = time_slices
        self.names = names

    def display(self):

        data = [
            [
                ("&nbsp;", None, False, False, "green"),
                ("– ranked better<br/>(statistically)", None, False, False, None),
            ],
            [
                ("&nbsp;", None, False, False, "darkorange"),
                ("– similar rank<br/>(statistically)", None, False, False, None),
            ],
            [
                ("*", None, False, False, None),
                (
                    "– not different from<br/>reference (statistically)",
                    None,
                    False,
                    False,
                    None,
                ),
            ],
        ]

        return EvalTable(data=data, title="Legend").as_html()
