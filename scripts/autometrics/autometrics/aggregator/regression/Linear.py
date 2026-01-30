from sklearn.linear_model import LinearRegression

from autometrics.aggregator.regression import Regression

class Linear(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        model = LinearRegression()

        if not name:
            name = "LinearRegression"

        if not description:
            description = "Linear regression"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
