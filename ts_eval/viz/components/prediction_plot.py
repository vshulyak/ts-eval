from base64 import b64encode

# TODO: maybe set matplotlib/IPython as deps... or catch ImportError
import matplotlib.pyplot as plt

from IPython.core.pylabtools import print_figure
from matplotlib._pylab_helpers import Gcf
from pandas.plotting import register_matplotlib_converters

from . import BaseComponent


class PredictionPlotComponent(BaseComponent):
    component_type = "plot"

    template = """
        <img src='data:image/png;base64,{point_graph_data}'>
    """

    def __init__(
        self,
        target,
        preds,
        points,
        time_slices,
        names,
        figsize,
        cm_name="Dark2",
        pi=True,
    ):
        self.target = target
        self.preds = preds
        self.points = points
        self.time_slices = time_slices
        self.names = names
        self.figsize = figsize
        self.cm = plt.get_cmap(cm_name)
        self.pi = pi

    def display(self):

        # modifies global matplotlib unit registry. A warning is raised
        # if this line doesn't exist. Probably, there's a better way to handle
        # this, but for now it's fine.
        register_matplotlib_converters()

        fig, ax = plt.subplots(
            len(self.points), 1, sharex=True, sharey=True, figsize=self.figsize
        )

        colors = self.cm(range(len(self.preds) + 1))

        for i, point in enumerate(self.points):

            t = self.target.mean_[:, point]
            ax[i].plot(self.target.dt, t, color=colors[0], linewidth=3, label="target")
            for pi, pr in enumerate(self.preds):
                p = pr.mean_.values[:, point]
                name = self.names[pi]
                pd_idx = pr.dt.to_index()
                ax[i].plot(
                    pd_idx,
                    p,
                    color=colors[pi + 1],
                    linestyle="--",
                    linewidth=2,
                    label=name,
                )

                # some datasets might have ub/lb missing
                ub = pr.upper.values[:, point] if hasattr(pr, "upper") else None
                lb = pr.lower.values[:, point] if hasattr(pr, "lower") else None

                if self.pi and ub is not None and lb is not None:
                    ax[i].fill_between(pd_idx, ub, lb, color=colors[pi + 1], alpha=0.1)

            if i == 0:
                ax[i].legend(loc="upper right")
            ax[i].set_title(f"horizon {point}", loc="right")
            ax[i].grid(True)
            ax[i].margins(x=0.01)
            # TODO: only if datetime index
        #             ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        #             ax[i].xaxis.set_minor_locator(mdates.DayLocator())
        #             monthFmt = mdates.DateFormatter('%b')
        #             ax[i].xaxis.set_major_formatter(monthFmt)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        Gcf.destroy_fig(fig)

        return self.template.format(
            point_graph_data=b64encode(print_figure(fig)).decode("utf-8")
        )
