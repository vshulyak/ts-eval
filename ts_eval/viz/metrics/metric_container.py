from collections import defaultdict

import numpy as np

from ts_eval.utils import nanmeanw
from ts_eval.viz.stats.mann_whitney_u import mw_is_equal
from ts_eval.viz.stats.rank_test import rank_test_2d, rank_test_3d
from ts_eval.viz.utils import filter_nan


class MetricResult:
    """
    Holds computed metrics and allows aggregations on top of them.

    TODO:
     - cache results
    """

    def __init__(self, ref, rest, relative, fv):
        self.ref = ref
        self.rest = rest
        self.relative = relative
        self.fv = fv

    def prediction_rankings(self):
        """
        Computes rank of predictions and if they are statistically different
        """
        overall_data = filter_nan(nanmeanw(self.rest, 1)).reshape(
            -1, self.rest.shape[2]
        )

        overall = rank_test_2d(overall_data)
        steps = rank_test_3d(self.rest)

        return overall, steps

    def ref_equality(self):
        """
        Computes equality with reference prediction (for relative metrics)
        """
        if self.ref is None:
            overall = np.full(self.rest.shape[2], False)
            steps = np.full((self.rest.shape[1], self.rest.shape[2]), False)
            return overall, steps

        overall = []
        steps = []
        for i in range(self.rest.shape[2]):
            overall += [
                mw_is_equal(
                    filter_nan(nanmeanw(self.ref, 1)),
                    filter_nan(nanmeanw(self.rest[:, :, i], 1)),
                )
            ]
            # TODO: it was for p in points, can I do it in one batch?
            step = [
                mw_is_equal(
                    filter_nan(self.ref[:, p]), filter_nan(self.rest[:, :, i][:, p])
                )
                for p in range(self.rest.shape[1])
            ]

            steps += [step]

        return np.stack(overall), np.stack(steps, 1)

    def steps(self):
        """
        Compute predictions on every step (horizon)
        """
        rest = nanmeanw(self.rest, 0)
        if self.relative:
            assert self.ref is not None, "Reference prediction should be provided"
            ref = nanmeanw(self.ref, 0).reshape(-1, 1)
            return 1 - rest / ref if self.fv else rest / ref
        return rest

    def overall(self):
        """
        Compute overall score across all steps

        Notes:
        - We don't have to store overall score as we can average steps to get the same result
        """
        return nanmeanw(self.steps(), 0)


class MetricContainer:
    """
    Holds all computed metrics and allows access to slices / metrics.
    """

    def __init__(self, metric_res):
        self.metric_res = metric_res

    @classmethod
    def build(cls, target, ref, preds, slices, metrics):

        metric_res = defaultdict(lambda: dict())

        for slice in slices:

            target_slice = slice(target)
            ref_slice = slice(ref) if ref is not None else None
            pred_slices = [slice(ds) for ds in preds]

            for metric in metrics:
                scores = []

                ref_score = (
                    metric(*metric.adaptor(target_slice, ref_slice))
                    if ref_slice is not None
                    else None
                )

                for pred_slice in pred_slices:
                    scores += [metric(*metric.adaptor(target_slice, pred_slice))]

                metric_res[slice][metric.name] = MetricResult(
                    ref=ref_score,
                    rest=np.stack(scores, 2),
                    relative=metric.relative,
                    fv=metric.fv,
                )

        return cls(metric_res)
