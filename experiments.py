"""Compare versions of the off-market model on the same months. From this folder:

    python experiments.py              run every variant below (about 15-20 minutes)
    python experiments.py v1 abs_loss  run only the variants named

Why separate tuning months? Every change we keep because it scored better on the
validation months (Dec 2025 to May 2026) makes that score a bit too optimistic: we
picked the winner on those very homes. So we compare variants on six EARLIER months
(June to November 2025), pick the best, and only then run `python run.py validate`
once, as a fair final check. The report can then say exactly that.

Prints a table and saves it to reports/experiments.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
from analysis import error_table
from models import OffMarketModel

ROOT = Path(__file__).parent
TUNING_MONTHS = pd.period_range("2025-06", "2025-11", freq="M")

# The feature groups added after the first version (v1, 9.33% validation MdAPE).
NEW_FEATURES = ["building_log_price", "building_log_price_per_sqft", "building_sales",
                "sized_comp_log_price", "sized_comp_distance",
                "market_change_3m", "market_change_12m"]

# Each variant changes ONE thing from "new_features", so we know what each change does.
VARIANTS = {
    "v1": lambda: OffMarketModel(exclude=NEW_FEATURES),               # the first version
    "new_features": lambda: OffMarketModel(),                         # + building, size-matched comps, momentum
    "abs_loss": lambda: OffMarketModel(loss="absolute_error"),        # aim at the median, like MdAPE
    "sized_guess": lambda: OffMarketModel(use_sized_comps=True),      # start from size-matched comps
    "k8": lambda: OffMarketModel(k=8),                                # fewer, closer comps
    "k25": lambda: OffMarketModel(k=25),                              # more comps, less noise
    "recent_weight": lambda: OffMarketModel(half_life_years=2),       # older sales count less
    "more_trees": lambda: OffMarketModel(max_iter=1500, learning_rate=0.03),
    "text": lambda: OffMarketModel(text_score=True),                  # + a score read from the remarks
    # Round 2: combinations of the round-1 winners, all with more_trees.
    "more_trees_text": lambda: OffMarketModel(max_iter=1500, learning_rate=0.03, text_score=True),
    "more_trees_sized": lambda: OffMarketModel(max_iter=1500, learning_rate=0.03, use_sized_comps=True),
    "most_trees": lambda: OffMarketModel(max_iter=3000, learning_rate=0.02),
}


def run_variant(name, listings):
    """Walk-forward on the tuning months, as run.py validate does on its months."""
    rows = []
    for month in TUNING_MONTHS:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        model = VARIANTS[name]().fit(train)
        predicted = model.predict(data.hide_answers(homes, "off_market"))
        rows.append(pd.DataFrame({"variant": name, "sold_price": homes["sold_price"],
                                  "predicted_price": np.round(predicted)}))
    return pd.concat(rows)


def main(names):
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variant(s): {', '.join(unknown)}. Choose from: {', '.join(VARIANTS)}")
    listings = data.load("listings")
    results = []
    for name in names:
        start = time.time()
        results.append(run_variant(name, listings))
        print(f"{name}: done in {time.time() - start:.0f} s")
    predictions = pd.concat(results)
    table = error_table(predictions, predictions["variant"]).reindex(names)
    if "v1" in table.index:
        table["mdape_vs_v1"] = (table["mdape"] - table.at["v1", "mdape"]).round(2)

    lines = [f"# Off-market experiments, tuning months {TUNING_MONTHS[0]} to {TUNING_MONTHS[-1]}", "",
             "| variant | " + " | ".join(table.columns) + " |", "|---|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples()]
    lines += ["", "Keep a change only if it lowers MdAPE clearly (by more than about 0.1),",
              "then combine the winners and run `python run.py validate` once."]
    path = ROOT / "reports" / "experiments.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
