from sklearn.ensemble import GradientBoostingRegressor

from autometrics.aggregator.regression import Regression

class GradientBoosting(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        model = GradientBoostingRegressor()

        if not name:
            name = "GradientBoosting"

        if not description:
            description = "GradientBoosting regression"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
