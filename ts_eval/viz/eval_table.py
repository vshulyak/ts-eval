from functools import partial

from statsmodels.iolib.table import Cell, SimpleTable, pad


FORMATTER_NAME = "callable_formatted"


class TSEvalCell(Cell):
    def format(self, width, output_format="txt", **fmt_dict):
        if self.datatype == FORMATTER_NAME:
            return self.format_string_formatted(
                width, output_format=output_format, **fmt_dict
            )
        else:
            return super().format(width, output_format=output_format, **fmt_dict)

    def format_string_formatted(self, width, output_format="txt", **fmt_dict):
        fmt = self._get_fmt(output_format, **fmt_dict)
        data = self.data
        datatype = self.datatype

        assert callable(fmt[datatype]), "A callable should be provided as formatter"

        content = fmt[datatype](data)
        content = self._latex_escape(content, fmt, output_format)

        align = self.alignment(output_format, **fmt)
        return pad(content, width, align)


def html_formatter(data):
    assert len(data) == 5, "5-item tuple is expected as input"
    value, color, warn_sign, is_text_bold, bg_color = data

    style = f"style='color: {color}'" if color else ""
    style = f"style='background: {bg_color}'" if bg_color else style
    warn_text = "*" if warn_sign else ""
    text = (
        f"<strong>{value}{warn_text}</strong>"
        if is_text_bold
        else f"{value}{warn_text}"
    )

    return f"<td {style}>{text}</td>"


def txt_formatter(data):
    assert len(data) == 5, "5-item tuple is expected as input"
    value, _, warn_sign, _, _ = data

    warn_text = "*" if warn_sign else ""
    return f"{value}{warn_text}"


EvalTable = partial(
    SimpleTable,
    datatypes=[FORMATTER_NAME],
    html_fmt={FORMATTER_NAME: html_formatter},
    txt_fmt={FORMATTER_NAME: txt_formatter},
    celltype=TSEvalCell,
)
