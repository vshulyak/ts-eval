from datetime import datetime
from operator import itemgetter

from statsmodels.iolib.table import SimpleTable

from ..utils import nphash
from . import BaseComponent


class DatasetDescriptionComponent(BaseComponent):
    component_type = "description"

    def __init__(
        self,
        target,
        pred,
        points,
        time_slice,
        date_format="%a, %d %b %Y",
        time_format="%H:%M:%S",
    ):
        self.target = target
        self.pred = pred
        self.points = points
        self.time_slice = time_slice
        self.date_format = date_format
        self.time_format = time_format

        self.data_container = None

    def display(self):

        now = datetime.now()

        data = [
            ("Date:", [now.strftime(self.date_format)]),
            ("Time:", [now.strftime(self.time_format)]),
            ("No. Timepoints:", [self.target.sizes["dt"]]),
            ("Horizon", [self.target.sizes["h"]]),
            ("Target Hash", [nphash(self.target.mean_.values)]),
        ]

        return SimpleTable(
            data=list(map(itemgetter(1), data)),
            stubs=list(map(itemgetter(0), data)),
            title="Dataset Description",
        ).as_html()
