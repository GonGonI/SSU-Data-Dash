"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X


# ---------------------------------------------------------------------------
# Off-market model (OffMarketModel in models.py)
#
# The features above describe the home itself. The ones below describe what similar
# homes nearby sold for. They need the table of past sales as well as the homes, and
# every one of them uses only sales from before the month the home was listed: the
# same rule for the training homes as for the homes we predict, so the model never
# learns from information it would not have in real use.
# ---------------------------------------------------------------------------

KM_PER_DEGREE = 111.2   # one degree of latitude; a degree of longitude is 0.72 of that here

# Words in the remarks that might signal condition or quality. Each becomes a 0/1 column;
# the trees ignore any that do not help. Read some remarks and add your own.
REMARK_PATTERNS = {
    "says_renovated": r"renovat|updated|upgraded|remodel|modernized",
    "says_needs_work": r"needs work|\btlc\b|handyman|contractor|as[- ]is|investor|fixer|potential",
    "says_luxury": r"luxur|high[- ]end|custom|designer|premium",
    "says_new": r"brand new|newly built|new build|never lived",
    "says_view": r"\bviews?\b|lake|ravine|skyline",
    "says_pool": r"\bpool\b",
}


def month_number(dates):
    """Months counted from year 0, so month arithmetic is plain integer arithmetic
    (May 2026 minus 12 is May 2025)."""
    return dates.dt.year * 12 + dates.dt.month - 1


def log_square_feet(house_area):
    """log of square_feet(), empty where the size is missing or zero."""
    sqft = square_feet(house_area)
    return np.log(sqft.where(sqft > 0))


def km_coordinates(table):
    """Latitude and longitude as x, y in kilometres, so distances are in km."""
    x = table["longitude"].to_numpy(dtype=float) * 0.72 * KM_PER_DEGREE
    y = table["latitude"].to_numpy(dtype=float) * KM_PER_DEGREE
    return np.column_stack([x, y])


def home_features(homes, categories):
    """What the home itself is like: the starter features plus lot size, garage, style,
    municipality and words in the remarks. `categories` holds the text values seen in
    training, so every text column is coded the same way for training and test homes."""
    X = off_market_features(homes)
    X["log_square_feet"] = log_square_feet(homes["house_area"])
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    X["garage"] = pd.to_numeric(homes["garage"], errors="coerce")
    for column, values in categories.items():
        X[column] = pd.Categorical(homes[column], categories=values)
    remarks = homes["remarks"].fillna("").str.lower()
    X["remarks_length"] = remarks.str.len()
    for name, pattern in REMARK_PATTERNS.items():
        X[name] = remarks.str.contains(pattern, regex=True).astype(int)
    return X


def market_index(sales, last_month):
    """The price level of each property type, month by month: the mean log sold price
    over the three months ending in that month.

    Returns a table with one row per month (as a month_number) and one column per
    property type. Months after the last sale repeat the last known level, so a home
    listed after the training data ends is valued at the latest prices.
    """
    month = month_number(sales["date_sold"])
    log_price = np.log(sales["sold_price"])
    grouped = log_price.groupby([month, sales["house_type_name"]])
    totals, counts = grouped.sum().unstack(), grouped.count().unstack()
    months = range(int(month.min()), max(int(month.max()), int(last_month)) + 1)
    totals = totals.reindex(months).fillna(0).rolling(3, min_periods=1).sum()
    counts = counts.reindex(months).fillna(0).rolling(3, min_periods=1).sum()
    return (totals / counts.where(counts > 0)).ffill()


COMPARABLE_COLUMNS = ["comp_log_price", "comp_log_price_per_sqft", "comp_log_square_feet",
                      "comp_spread", "comp_distance_km", "comp_months_ago",
                      "building_log_price", "building_log_price_per_sqft", "building_sales",
                      "sized_comp_log_price", "sized_comp_distance",
                      "community_log_price", "community_sales",
                      "market_level", "market_change_3m", "market_change_12m"]

SAME_BUILDING_KM = 0.05   # comps closer than 50 m count as the same building or block


def comparable_features(homes, sales, k=15, comp_years=2, community_months=12, size_weight=5.0):
    """What similar homes sold for, from sales before the month each home was listed.

    For each home, using sales of the same property type only:
    - comp_*: its k nearest earlier sales within the last `comp_years` years. Each comp's
      log price is first moved to today's market level with market_index(), so a sale
      from 18 months ago counts at what it would fetch now. comp_log_price is the
      median of those adjusted log prices.
    - community_*: the median adjusted log price of the same type in the same
      community over the last `community_months` months, and how many sales that is.
      (Close to the 17.7% "maximum error" baseline from the handout.)
    - building_*: those of the k comps within 50 m: in a condo building, the sales of
      other units in the same building, usually the best comparables there are.
    - sized_comp_*: the k nearest comps when size counts as well as location: each
      comp's distance adds `size_weight` km per unit of log square feet, so a comp
      twice the size counts as 3.5 km further away. Helps for homes much bigger or
      smaller than their neighbours.
    - market_level: the type's market index in the month before listing, and
      market_change_*: how much it moved over the 3 and 12 months before that (a
      rising market tends to sell above its recent comps).
    """
    out = pd.DataFrame(np.nan, index=homes.index, columns=COMPARABLE_COLUMNS)
    listed = month_number(homes["date_listed"])
    index = market_index(sales, last_month=listed.max())

    sale_month = month_number(sales["date_sold"]).to_numpy()
    log_price = np.log(sales["sold_price"].to_numpy(dtype=float))
    log_sqft = log_square_feet(sales["house_area"]).to_numpy()
    kinds = sales["house_type_name"].to_numpy()
    communities = sales["community"].to_numpy()
    sale_xy = km_coordinates(sales)
    home_xy = km_coordinates(homes)
    positions = pd.Series(np.arange(len(homes)), index=homes.index)

    # One group per property type and listing month: they all share the same pool of
    # earlier sales, so one search tree serves the whole group.
    for (kind, month), group in homes.groupby([homes["house_type_name"], listed]):
        if kind not in index.columns:
            continue
        reference = month - 1   # the last full month before listing
        if reference not in index.index or np.isnan(index.at[reference, kind]):
            continue
        rows = positions[group.index].to_numpy()
        level = index[kind]
        out.loc[group.index, "market_level"] = level.at[reference]
        for months_back in (3, 12):
            if reference - months_back in level.index:
                change = level.at[reference] - level.at[reference - months_back]
                out.loc[group.index, f"market_change_{months_back}m"] = change

        earlier = (kinds == kind) & (sale_month < month)
        level_then = index[kind].reindex(sale_month[earlier]).to_numpy()
        adjusted = log_price[earlier] + index.at[reference, kind] - level_then

        # Same type, same community, last `community_months` months.
        recent = sale_month[earlier] >= month - community_months
        by_community = pd.Series(adjusted[recent]).groupby(communities[earlier][recent])
        out.loc[group.index, "community_log_price"] = group["community"].map(by_community.median()).to_numpy()
        out.loc[group.index, "community_sales"] = group["community"].map(by_community.size()).fillna(0).to_numpy()

        # The k nearest sales of the same type within the last `comp_years` years.
        window = sale_month[earlier] >= month - 12 * comp_years
        if window.sum() == 0:
            continue
        pool_xy = sale_xy[earlier][window]
        pool_price = adjusted[window]
        pool_sqft = log_sqft[earlier][window]
        pool_age = month - sale_month[earlier][window]
        n = min(k, len(pool_price))
        distance, nearest = cKDTree(pool_xy).query(home_xy[rows], k=n)
        distance, nearest = distance.reshape(len(rows), n), nearest.reshape(len(rows), n)

        prices = pool_price[nearest]
        sqft = pool_sqft[nearest]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # nanmedian of comps with no sizes
            out.loc[group.index, "comp_log_price"] = np.median(prices, axis=1)
            out.loc[group.index, "comp_log_price_per_sqft"] = np.nanmedian(prices - sqft, axis=1)
            out.loc[group.index, "comp_log_square_feet"] = np.nanmedian(sqft, axis=1)
            out.loc[group.index, "comp_spread"] = np.std(prices, axis=1)
            out.loc[group.index, "comp_distance_km"] = np.median(distance, axis=1)
            out.loc[group.index, "comp_months_ago"] = np.median(pool_age[nearest], axis=1)

            same_building = distance < SAME_BUILDING_KM
            out.loc[group.index, "building_sales"] = same_building.sum(axis=1)
            out.loc[group.index, "building_log_price"] = np.nanmedian(
                np.where(same_building, prices, np.nan), axis=1)
            out.loc[group.index, "building_log_price_per_sqft"] = np.nanmedian(
                np.where(same_building, prices - sqft, np.nan), axis=1)

        # The k nearest by location and size together, for homes and comps with a size.
        home_sqft = log_square_feet(group["house_area"]).to_numpy()
        sized_pool = np.isfinite(pool_sqft)
        sized_homes = np.isfinite(home_sqft)
        if sized_pool.sum() == 0 or sized_homes.sum() == 0:
            continue
        pool_xyz = np.column_stack([pool_xy[sized_pool], size_weight * pool_sqft[sized_pool]])
        home_xyz = np.column_stack([home_xy[rows][sized_homes], size_weight * home_sqft[sized_homes]])
        n = min(k, int(sized_pool.sum()))
        distance, nearest = cKDTree(pool_xyz).query(home_xyz, k=n)
        distance, nearest = distance.reshape(-1, n), nearest.reshape(-1, n)
        sized_index = group.index[sized_homes]
        out.loc[sized_index, "sized_comp_log_price"] = np.median(pool_price[sized_pool][nearest], axis=1)
        out.loc[sized_index, "sized_comp_distance"] = np.median(distance, axis=1)
    return out


def starting_guess(comparables, use_sized_comps=False):
    """A first estimate of each home's log price before the model's corrections: its
    comparables if it has them (matched on size too, if use_sized_comps), else its
    community, else its property type's level."""
    guess = comparables["comp_log_price"]
    if use_sized_comps:
        guess = comparables["sized_comp_log_price"].fillna(guess)
    return (guess
            .fillna(comparables["community_log_price"])
            .fillna(comparables["market_level"]))


def off_market_model_features(homes, sales, categories, use_sized_comps=False, **comp_settings):
    """Every feature of the off-market model: the home itself, what similar homes sold
    for, and how this home differs from them (bigger than its comps? that matters).
    comp_settings (k, comp_years, size_weight) go to comparable_features()."""
    X = home_features(homes, categories)
    comparables = comparable_features(homes, sales, **comp_settings)
    X = X.join(comparables)
    X["starting_guess"] = starting_guess(comparables, use_sized_comps)
    X["sqft_vs_comps"] = X["log_square_feet"] - X["comp_log_square_feet"]
    X["guess_from_price_per_sqft"] = X["comp_log_price_per_sqft"] + X["log_square_feet"]
    return X
