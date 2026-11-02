"""Error analysis for the report. From this folder:

    python analysis.py breakdown   where the off-market model misses: by property type,
                                   by month and by price range (needs `python run.py validate` first)
    python analysis.py horizon     how the error grows when the model has to predict
                                   further past its training data (takes about 15 minutes)
    python analysis.py horizon-figure
                                   redraw figures/error_by_horizon.png from the saved
                                   results, without rerunning the models

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
    table = error_table(predictions, predictions["gap"])
    tables = {"Months between the end of the training data and the sale": table}
    write_tables("Off-market error by forecast horizon", tables, ROOT / "reports" / "error_by_horizon.md")
    table.rename_axis("gap").to_csv(ROOT / "reports" / "error_by_horizon.csv")
    horizon_figure()


# The handout's accuracy bar: the simple baseline's MdAPE on validation.
ACCURACY_BAR = 17.7


def horizon_figure():
    """Figure: off-market MdAPE by months between the end of the training data and the
    sale, from reports/error_by_horizon.csv (written by `python analysis.py horizon`)."""
    import matplotlib.pyplot as plt

    path = ROOT / "reports" / "error_by_horizon.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python analysis.py horizon` first")
    table = pd.read_csv(path)
    ink, quiet, grid, line = "#0b0b0b", "#52514e", "#e5e5e5", "#2a78d6"

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(table["gap"], table["mdape"], color=line, linewidth=2, marker="o", markersize=7, zorder=3)
    # Label the first and last point: where validation stands, and the furthest test homes.
    for i in (0, len(table) - 1):
        gap, mdape = table.loc[i, "gap"], table.loc[i, "mdape"]
        ax.annotate(f"{mdape:.2f}%", (gap, mdape), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, color=ink, fontweight="bold")
    ax.axhline(ACCURACY_BAR, color=quiet, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(table["gap"].max(), ACCURACY_BAR + 0.6, f"Accuracy bar: {ACCURACY_BAR}% (simple baseline)",
            ha="right", fontsize=9, color=quiet)
    ax.text(table["gap"].max(), 2.2, "Test homes sold Jun to Sep 2026 from data ending May 2026, 0 to 3 months\n"
            f"past the data: we expect a hidden-test MdAPE of about {table['mdape'].min():.1f}% to "
            f"{table['mdape'].max():.1f}%.", ha="right", fontsize=8.5, color=quiet)

    ax.set_xticks(table["gap"], [f"{g} month{'s' if g != 1 else ''}" for g in table["gap"]])
    ax.set_xlabel("Months between the end of the training data and the sale", color=quiet)
    ax.set_ylabel("Off-market MdAPE (%)", color=quiet)
    ax.set_ylim(0, 20)
    ax.set_xlim(table["gap"].min() - 0.3, table["gap"].max() + 0.3)
    ax.grid(axis="y", color=grid)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(grid)
    ax.tick_params(colors=quiet)
    ax.set_title("Predicting further ahead costs little: error rises from "
                 f"{table['mdape'].iloc[0]:.2f}% to {table['mdape'].iloc[-1]:.2f}%",
                 loc="left", fontsize=12, color=ink, fontweight="bold")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_horizon.png", dpi=200)
    plt.close(fig)
    print("Wrote figures/error_by_horizon.png")


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


COMMANDS = {"breakdown": breakdown, "horizon": horizon, "horizon-figure": horizon_figure}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]]()
