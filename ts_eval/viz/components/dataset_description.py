from datetime import datetime
from operator import itemgetter

from statsmodels.iolib.table import SimpleTable

from ..utils import nphash


class DatasetDescriptionComponent(object):
    component_type = "description"

    def __init__(
        self,
        target,
        pred,
        points,
        time_slice,
        date_format="%a, %d %b %Y",
        time_format="%H:%M:%S",
        hash_base=36,
    ):
        self.target = target
        self.pred = pred
        self.points = points
        self.time_slice = time_slice
        self.date_format = date_format
        self.time_format = time_format
        self.hash_base = hash_base

    def _repr_html_(self):

        now = datetime.now()

        data = [
            ("Date:", [now.strftime(self.date_format)]),
            ("Time:", [now.strftime(self.time_format)]),
            ("No. Timepoints:", [self.target.sizes["dt"]]),
            ("Horizon", [self.target.sizes["h"]]),
            ("Target Hash", [nphash(self.target.mean_)]),
        ]

        return SimpleTable(
            data=list(map(itemgetter(1), data)),
            stubs=list(map(itemgetter(0), data)),
            title="Dataset Description",
        ).as_html()
