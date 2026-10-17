"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListPriceBaseline()
        # return FeatureModel(features.list_aware_features)   # try this instead
    return OffMarketModel()
    # return NearestSalesBaseline(k=10)                       # the starter baseline
    # return FeatureModel(features.off_market_features)       # the starter's tree model


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class OffMarketModel:
    """Off-market model: start from what comparable homes sold for, then let
    gradient-boosted trees correct that guess for how this home differs.

    1. Starting guess: the median price of the 15 nearest earlier sales of the same
       type, each moved to the current market level (features.comparable_features).
    2. The trees learn log(sold price) - log(starting guess): how far above or below
       its comps a home sells, given its size, lot, location, remarks and so on.
    3. Prediction = starting guess * exp(the trees' correction).

    Why not let the trees predict the price directly? Trees cannot extrapolate: they
    never predict a price level they have not seen. The starting guess carries the
    market level forward in time; the trees only need to learn corrections, which
    change much more slowly than prices do.

    Every feature uses only sales from before the month the home was listed, for the
    training homes too, so the model learns under the same conditions it is used in.
    """

    CATEGORY_COLUMNS = ["house_style", "municipality"]
    SALE_COLUMNS = ["ml_num", "house_type_name", "community", "date_sold", "latitude", "longitude",
                    "house_area", "sold_price"]

    def __init__(self, **tree_settings):
        settings = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                        l2_regularization=1.0, random_state=0)
        settings.update(tree_settings)
        self.model = HistGradientBoostingRegressor(**settings)

    def features(self, homes):
        return features.off_market_model_features(homes, self.sales, self.categories)

    def fit(self, train):
        self.sales = train[self.SALE_COLUMNS]
        self.categories = {c: sorted(train[c].dropna().unique()) for c in self.CATEGORY_COLUMNS}
        X = self.features(train)
        target = np.log(train["sold_price"]) - X["starting_guess"]
        # The first homes of 2021 have no earlier sales to compare with: skip them.
        usable = target.notna()
        self.fallback = float(np.log(train["sold_price"]).median())
        self.model.fit(X[usable], target[usable])
        return self

    def predict(self, homes):
        X = self.features(homes)
        guess = X["starting_guess"].fillna(self.fallback)
        return np.exp(guess + self.model.predict(X)).to_numpy()
