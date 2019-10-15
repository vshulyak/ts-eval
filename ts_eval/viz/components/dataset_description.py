from datetime import datetime
from operator import itemgetter

import numpy as np

from statsmodels.iolib.table import SimpleTable

from ..utils import nphash
from . import BaseComponent


class DatasetDescriptionComponent(BaseComponent):
    component_type = "description"

    def __init__(
        self,
        target,
        preds,
        points,
        time_slices,
        names,
        ref_name,
        date_format="%a, %d %b %Y",
        time_format="%H:%M:%S",
    ):
        self.target = target
        self.preds = preds
        self.points = points
        self.time_slices = time_slices
        self.names = names
        self.ref_name = ref_name
        self.date_format = date_format
        self.time_format = time_format

    def display(self):

        now = datetime.now()

        pred_hashes = []
        for pred in self.preds:

            pred_data = np.stack(
                [
                    getattr(pred, a).values
                    for a in ["upper", "mean_", "lower"]
                    if hasattr(pred, a)
                ],
                2,
            )
            pred_hashes += [nphash(pred_data)]

        data = [
            ("Date:", [now.strftime(self.date_format)]),
            ("Time:", [now.strftime(self.time_format)]),
            ("No. Timepoints:", [self.target.sizes["dt"]]),
            ("Horizon", [self.target.sizes["h"]]),
        ]

        if self.ref_name:
            data += [(f"Reference Metric", [self.ref_name])]

        data += [("Target Hash", [nphash(self.target.mean_.values)])]

        for name, pred_hash in zip(self.names, pred_hashes):
            # done explicitly to make it clear
            data += [(f'"{name}" Hash', [pred_hash])]

        if self.ref_name:
            data += [(f"Reference Metric Hash", [pred_hash])]

        return SimpleTable(
            data=list(map(itemgetter(1), data)),
            stubs=list(map(itemgetter(0), data)),
            title="Dataset Description",
        ).as_html()
