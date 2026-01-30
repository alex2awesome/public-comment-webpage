from sklearn.linear_model import Lasso as Las

from autometrics.aggregator.regression import Regression

class Lasso(Regression):
    def __init__(self, name=None, description=None, dataset=None, alpha=0.01, **kwargs):
        model = Las(alpha=alpha)

        if not name:
            name = "Lasso"

        if not description:
            description = f"Lasso regression (alpha={alpha})"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
