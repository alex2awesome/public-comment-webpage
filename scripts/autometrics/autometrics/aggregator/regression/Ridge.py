from sklearn.linear_model import Ridge as Rid

from autometrics.aggregator.regression import Regression

class Ridge(Regression):
    def __init__(self, name=None, description=None, dataset=None, alpha=0.01, **kwargs):
        model = Rid(alpha=alpha)

        if not name:
            name = "Ridge"

        if not description:
            description = f"Ridge regression (alpha={alpha})"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
