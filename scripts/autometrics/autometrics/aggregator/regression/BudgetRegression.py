from autometrics.aggregator.regression import Regression
from autometrics.metrics.MultiMetric import MultiMetric
from sklearn.preprocessing import StandardScaler

class BudgetRegression(Regression):
    """
    Class for regression aggregation with metric budget constraint
    """
    def __init__(self, regressor: Regression, metric_budget: int, name=None, description=None, dataset=None, **kwargs):
        """
        Initialize the class
        """
        self.regressor = regressor
        self.model = regressor.model
        self.name = regressor.name + f"Top {metric_budget}" if not name else name
        self.description = regressor.description + f"Top {metric_budget}" if not description else description
        self.dataset = regressor.dataset if not dataset else dataset
        self.metric_budget = metric_budget  

        super().__init__(self.name, self.description, model=self.model, dataset=self.dataset, **kwargs)

    def learn(self, dataset, target_column=None):
        """
        Learn the regression model with proper scaling and metric selection
        """
        self.ensure_dependencies(dataset)

        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        X = df[input_columns]
        y = df[target_column]

        # Apply StandardScaler to handle scale differences between metrics
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model on scaled data
        self.model.fit(X_scaled, y)

        # Get important metrics using standardized coefficients
        important_metrics = self.identify_important_metrics()[:self.metric_budget]
        important_metrics = [metric[1] for metric in important_metrics]

        # Filter input metrics to only include the top ones
        self.input_metrics = [metric for metric in self.input_metrics if (not isinstance(metric, MultiMetric) and metric.get_name() in important_metrics) or (isinstance(metric, MultiMetric) and any(submetric in important_metrics for submetric in metric.get_submetric_names()))]

        # Refit model on selected metrics only (with scaling)
        selected_columns = self.get_input_columns()
        X_selected = df[selected_columns]
        X_selected_scaled = self.scaler.transform(X_selected)
        self.model.fit(X_selected_scaled, y)
