"""Error analysis for the report. From this folder:

    python analysis.py breakdown   where the off-market model misses: by property type,
                                   by month and by price range (needs `python run.py validate` first)
    python analysis.py horizon     how the error grows when the model has to predict
                                   further past its training data (takes a few minutes)

Both print tables and save them to reports/ so you can copy them into the report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
from run import VALIDATION_MONTHS

ROOT = Path(__file__).parent


def error_table(predictions, group):
    """One row per group: how many homes, MdAPE, MAPE, and the median signed error
    (above 0: the model predicts too high for that group)."""
    signed = 100 * (predictions["predicted_price"] - predictions["sold_price"]) / predictions["sold_price"]
    errors = pd.DataFrame({"group": group, "signed": signed, "absolute": signed.abs()})
    table = errors.groupby("group", observed=True).agg(
        homes=("absolute", "size"),
        mdape=("absolute", "median"),
        mape=("absolute", "mean"),
        median_signed_error=("signed", "median"),
    )
    return table.round(2)


def breakdown():
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path)
    off_market = predictions[predictions["track"] == "off_market"]

    # Price range: split the homes into ten equal groups by sold price (1 = cheapest tenth).
    decile = pd.qcut(off_market["sold_price"], 10, labels=False) + 1
    tables = {
        "Property type": error_table(off_market, off_market["house_type_name"]),
        "Month sold": error_table(off_market, off_market["month"]),
        "Price decile (1 = cheapest tenth)": error_table(off_market, decile),
    }
    write_tables("Off-market error breakdown", tables, ROOT / "reports" / "error_breakdown.md")


def horizon(gaps=(0, 1, 2, 3)):
    """Walk-forward validation again, but leaving a gap between the end of the
    training data and the month predicted.

    Gap 0 is what `run.py validate` does: train on everything before the month.
    Gap 3 trains on sales up to 3 months before the month, so the model predicts
    3 months past its data, like the September homes in the real test (trained to May).
    """
    listings = data.load("listings")
    rows = []
    for gap in gaps:
        for month in VALIDATION_MONTHS:
            train = listings[listings["date_sold"] < (month - gap).start_time]
            homes = listings[listings["date_sold"].dt.to_period("M") == month]
            model = models.make_model("off_market").fit(train)
            predicted = model.predict(data.hide_answers(homes, "off_market"))
            rows.append(pd.DataFrame({"gap": gap, "sold_price": homes["sold_price"],
                                      "predicted_price": predicted}))
            print(f"gap {gap} months, {month}: trained on sales before {month - gap}")
    predictions = pd.concat(rows)
    tables = {"Months between the end of the training data and the sale":
              error_table(predictions, predictions["gap"])}
    write_tables("Off-market error by forecast horizon", tables, ROOT / "reports" / "error_by_horizon.md")


def write_tables(title, tables, path):
    lines = [f"# {title}", ""]
    for name, table in tables.items():
        lines += [f"## {name}", "",
                  "| " + " | ".join([table.index.name or "", *table.columns]) + " |",
                  "|" + "---|" * (len(table.columns) + 1)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples()]
        lines.append("")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {path.relative_to(ROOT)}")


COMMANDS = {"breakdown": breakdown, "horizon": horizon}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]]()
