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
        ref,
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
        self.ref = ref
        self.ref_name = ref_name
        self.date_format = date_format
        self.time_format = time_format

    def display(self):

        now = datetime.now()
        pred_hashes = [self._build_xarray_hash(pred) for pred in self.preds]

        data = [
            ("Date:", [now.strftime(self.date_format)]),
            ("Time:", [now.strftime(self.time_format)]),
            ("No. Timepoints:", [self.target.sizes["dt"]]),
            ("Horizon", [self.target.sizes["h"]]),
        ]

        if self.ref_name:
            data += [(f"Reference Metric", [self.ref_name])]

        data += [("Target Hash", [self._build_xarray_hash(self.target)])]

        for name, pred_hash in zip(self.names, pred_hashes):
            # done explicitly to make it clear
            data += [(f'"{name}" Hash', [pred_hash])]

        if self.ref is not None:
            data += [(f"Reference Metric Hash", [self._build_xarray_hash(self.ref)])]

        return SimpleTable(
            data=list(map(itemgetter(1), data)),
            stubs=list(map(itemgetter(0), data)),
            title="Dataset Description",
        ).as_html()

    def _build_xarray_hash(self, xarr):
        arr = np.stack(
            [
                getattr(xarr, a).values
                for a in ["upper", "mean_", "lower"]
                if hasattr(xarr, a)
            ],
            2,
        )
        return nphash(arr)
