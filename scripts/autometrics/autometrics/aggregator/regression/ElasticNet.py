from sklearn.linear_model import ElasticNet as ENet

from autometrics.aggregator.regression import Regression

class ElasticNet(Regression):
    def __init__(self, name=None, description=None, dataset=None, alpha=0.01, l1_ratio=0.5, **kwargs):
        model = ENet(alpha=alpha, l1_ratio=l1_ratio)

        if not name:
            name = "ElasticNet"

        if not description:
            description = f"ElasticNet regression (alpha={alpha}, l1_ratio={l1_ratio})"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
