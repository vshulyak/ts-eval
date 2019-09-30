class JupyterHTMLLayout(object):
    # Actually, this can override styles out of scope of this component.
    # Pandas, for instance, generates unique IDs to cope with that.
    # But it's ok to keep it this way for now.

    style = """
    <style type="text/css" >
        .rendered_html .tsmetrics td {
            vertical-align: top;
            white-space: nowrap;
        }

        .rendered_html .tsmetrics td.graph-area {
            padding: 0;
        }

        .rendered_html .tsmetrics .row {
            text-align: left;
            white-space: normal;
        }

        .rendered_html .tsmetrics .row table {
            display: inline-block;
            margin: 0 2em 2em 0;
        }
    </style>
    """

    layout_with_right_panel = """
    <table class="tsmetrics">
        <tr style="background: none; padding: 0">
            <td class="table-area">
                {left_plot_area}
            </td>
            <td class="graph-area">
                {right_plot_area}
            </td>
        </tr>
        <tr><td class="row" colspan="2">{footer}</td></tr>
        <tr><td class="row" colspan="2"></td></tr>
    </table>
    """

    def __init__(self, components):
        self.components = components

    def _repr_html_(self):

        # TODO: should be customizable
        template = self.layout_with_right_panel

        left_plot_area = []
        right_plot_area = []
        footer = []

        for c in self.components.values():
            if c.component_type == "description":
                left_plot_area.append(c.display())
            elif c.component_type == "plot":
                right_plot_area.append(c.display())
            else:
                footer.append(c.display())

        return self.style + template.format(
            **{
                "left_plot_area": "".join(left_plot_area),
                "right_plot_area": "".join(right_plot_area),
                "footer": "".join(footer),
            }
        )
