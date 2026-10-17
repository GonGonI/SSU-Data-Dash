# Off-market model: how it works and how to work on it

This is a walkthrough of the code, in the order it runs, written for the team.

## 1. Get this branch into VS Code

The work is on the branch `claude/keen-wright-6wgzt7` of `github.com/GonGonI/SSU-Data-Dash`.

In the VS Code terminal (`` Ctrl+` ``), inside your `SSU-Data-Dash` folder:

```
git remote -v                       # which GitHub repo is "origin"?
```

If origin is `GonGonI/SSU-Data-Dash`:

```
git fetch origin claude/keen-wright-6wgzt7
git checkout claude/keen-wright-6wgzt7
```

If origin is `erwinzhang7/SSU-Data-Dash` (you cloned the starter), add the team repo first:

```
git remote add team https://github.com/GonGonI/SSU-Data-Dash
git fetch team claude/keen-wright-6wgzt7
git checkout -b claude/keen-wright-6wgzt7 team/claude/keen-wright-6wgzt7
```

To see what changed:
- **Source Control panel** (left sidebar, branch icon), or the branch name at the bottom-left, to
  switch branches.
- `git log --oneline -5` lists the commits; `git diff main -- features.py models.py` shows the code
  changes. In VS Code, open the command palette (`Cmd+Shift+P`) and run **Git: Open Changes**, or
  click a file in the Source Control panel, to get a side-by-side diff.

Then run it:

```
source .venv/bin/activate
pytest                      # 20 quick tests on made-up data, a few seconds
python run.py validate      # about a minute: 6 months x (fit + predict)
python run.py report        # the MdAPE table you log
```

## 2. How the pipeline flows

```
run.py validate
  for each month Dec 2025 ... May 2026:
      train = sales before that month              <- no future data
      homes = sales in that month
      model = models.make_model(track)             <- models.py picks the model
      model.fit(train)
      model.predict(data.hide_answers(homes, track))   <- strips sold_price, list_price, ...
  -> outputs/validation_predictions.csv
run.py report  -> reports/validation_report.md, results_table.tex, figures/
```

Each model is a class with two methods, `fit(train)` and `predict(homes)`. To try a new model,
write a class with those two methods and return it from `make_model()`. `run.py` doesn't need
to change.

## 3. The off-market model (`OffMarketModel` in models.py)

The idea is the one an appraiser uses: find comparable sales, adjust for the market and for
differences, then correct.

**Step 1: starting guess from comparables** (`features.comparable_features`)

For each home, take sales of the **same property type** that sold **before the month the home was
listed**:

1. **Market index** (`features.market_index`): for each type and month, the mean log sold price over
   the last 3 months. It tracks the 2021–22 boom and the correction since.
2. **Time-adjust every past sale**: `adjusted log price = log price + index(month before listing) -
   index(month it sold)`. A condo that sold for $700k when the market was 5% higher counts as about
   $665k now.
3. **15 nearest comps** within the last 2 years (a KD-tree on km coordinates). The median of their
   adjusted log prices is `comp_log_price`, the starting guess. There are also comp price per sq ft,
   comp size, spread, distance and age.
4. **Community level**: the median adjusted price of the same type in the same community over the
   last 12 months. This is close to the handout's 17.7% baseline, used here as a feature.

**Step 2: trees learn the correction**

Target = `log(sold_price) - starting_guess`, so the trees learn questions like "how much more than
its comps does a home with 30% more sq ft, a garage and 'renovated' in the remarks sell for?" The
features are in `features.home_features`: size, beds, baths, parking, lot, garage, style,
municipality, remark keywords, location and time, plus `sqft_vs_comps`.

**Why the residual design?** Gradient-boosted trees can't extrapolate. Trained up to May 2026, they
can never predict a price level they haven't seen, so a direct price model goes stale over June to
September. The comps carry the current price level, and the trees only learn corrections, which
are stable over time.

**Why log?** MdAPE is a percentage error. In log space a 10% miss is the same size on a $400k condo
and a $3M house.

**Leakage rules this obeys** (and the tests in `tests/test_starter.py` check):
- every history feature uses only sales before the listing month, for training homes too;
- `list_price` is never used (the off-market test file doesn't have it);
- `external_avm_estimate` is never used.

## 4. Your workflow for experiments

One change at a time, then:

```
python run.py validate && python run.py report
```

and log a row in the shared experiment log: date, who, change, off-market MdAPE, list-aware MdAPE.

Where the knobs are:

| Knob | Where | Try |
|---|---|---|
| number of comps | `comparable_features(k=15)` | 8, 25 |
| comp time window | `comp_years=2` | 1, 3 |
| tree loss | `OffMarketModel(loss="absolute_error")` | it targets the median, which may suit MdAPE |
| tree size / speed | `max_iter`, `learning_rate`, `min_samples_leaf` in `OffMarketModel.__init__` | |
| remark words | `REMARK_PATTERNS` in features.py | read 50 remarks first |
| training years | filter `train` in `OffMarketModel.fit` | 2023+ only |

To pass settings in from `make_model()`: `return OffMarketModel(loss="absolute_error")`.

## 5. Honesty notes for the report

- **Don't over-tune on Dec–May.** Each extra try you pick by validation score makes that score more
  optimistic. Freeze the model on Nov 1 as planned, and say in the report how many variants you tried.
- **Horizon gap.** In validation a home is predicted at most about 2 months after the training data
  ends. The test homes sold June to September, up to 4 months after. The market index stays flat
  after May, so if prices move over the summer the hidden error will be somewhat higher. A good
  check for your section: validate with a gap (train up to month m-3, predict month m) and report
  both numbers.
- **Error analysis.** `validation_predictions.csv` has `house_type_name` and `month`, so group the
  errors by them (see `error_by_price_range` in run.py for a template). Off-market errors will likely
  be largest for detached houses and the top price decile, where comps are rare and land drives price.
