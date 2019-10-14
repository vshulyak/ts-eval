from ts_eval.viz.eval_table import EvalTable


DATA = [
    [(11.00, None, None, True), (12.123123, None, None, True)],
    [(11.00, "red", True, False), (12.123123, "yellow", None, False)],
    [(11.00, "cyan", None, False), (12.123123, "pink", True, False)],
]
HEADERS = ["mse", "mae"]
STUBS = ["0", "1", "2"]


def test_eval_table_smoke__html():

    gen = EvalTable(data=DATA, headers=HEADERS, stubs=STUBS, title="TS Eval").as_html()

    assert len(gen) > 0


def test_eval_table_smoke__txt():

    gen = EvalTable(data=DATA, headers=HEADERS, stubs=STUBS, title="TS Eval").as_text()

    assert len(gen) > 0
