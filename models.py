"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        #return ListPriceBaseline()
        return FeatureModel(features.list_aware_features)   # try this instead
    #return NearestSalesBaseline(k=10)
    return FeatureModel(features.off_market_features)       # try this instead


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
        # self.make_features = make_features
        # self.model = HistGradientBoostingRegressor(
        #     # 2. Decreased Learning Rate to increase accuracy and careful learning 
        #     learning_rate=0.05, #default is 0.1, lowering learning rate slows learning down but allows for more accuracy 
        #     max_iter=500, #default iteration is 100, but as learning
        #     # 3. changing loss function to be based on absolute error  
        #     loss= "absolute_error",
        #     random_state=0)
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(
            categorical_features="from_dtype", 
            learning_rate=0.05, 
            max_iter=500,
            # loss= "absolute_error",
            random_state=0)   # edit: added the argument
        self.categories = {}  
    
    def _encode(self, X, fit=False):
        X = X.copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                if fit:
                    self.categories[col] = list(X[col].value_counts().index[:250])
                X[col] = pd.Categorical(X[col], categories=self.categories[col])
        return X

    def fit(self, train):
        # 4. added sample weights -> adjusting weights to be heavier in more recent dates
        # age_months = (train["date_sold"].max() - train["date_sold"]).dt.days /30.44
        # weights = 0.5 ** (age_months /6)
        # self.model.fit(self.make_features(train), np.log(train["sold_price"]), sample_weight= weights)
        # self.model.fit(self.make_features(train), np.log(train["sold_price"])) # default weights
        self.model.fit(self._encode(self.make_features(train), fit=True), np.log(train["sold_price"])) #encode weights
        # self.model.fit(self._encode(self.make_features(train), fit=True), np.log(train["sold_price"]), sample_weight= weights) #encode +sample weights

        return self

    def predict(self, homes):
        # return np.exp(self.model.predict(self.make_features(homes)))
        return np.exp(self.model.predict(self._encode(self.make_features(homes))))

