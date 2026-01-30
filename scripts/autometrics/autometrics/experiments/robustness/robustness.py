# File: robustness/robustness.py

import hashlib
import pandas as pd
import dspy
import os

from autometrics.experiments.experiment import Experiment
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.experiments.results import TabularResult
from autometrics.experiments.robustness.perturb import ProducePerturbations
from autometrics.experiments.robustness.analysis import analyze_and_plot
from autometrics.aggregator.Aggregator import Aggregator

class RobustnessExperiment(Experiment):

    def _produce_perturbation_scores(self, dataset, perturbations):
        # Only use "obvious" mode; ignore subtle perturbations entirely
        worse_obvious, same_obvious, strategies = (
            perturbations["perturbed_worse_obvious"],
            perturbations["perturbed_same_obvious"],
            perturbations["strategies"],
        )

        inputs = dataset.get_dataframe()[dataset.get_input_column()].tolist()
        reference_columns = dataset.get_reference_columns()

        inputs_structured = [[inputs[i]] * len(strategies) for i in range(len(inputs))]
        inputs_structured = [item for sublist in inputs_structured for item in sublist]

        # Build the initial table ONLY for the perturbed examples present now:
        # worse_obvious (len(inputs)*len(strategies)) and same_obvious (len(inputs))
        # We'll append the original examples later (to keep lengths aligned during batch eval)
        data = {
            "input": inputs_structured + inputs,
            "model_output": [],
            "strategy": (strategies * len(worse_obvious))
                        + (["same_obvious"] * len(same_obvious)),
            "group": ["worse_obvious"] * len(worse_obvious) * len(strategies)
                     + ["same_obvious"] * len(same_obvious),
        }

        for ref_col in reference_columns:
            ref_values = dataset.get_dataframe()[ref_col].tolist()
            ref_values_structured = [[ref_values[i]] * len(strategies) for i in range(len(ref_values))]
            ref_values_structured = [item for sublist in ref_values_structured for item in sublist]
            data[ref_col] = ref_values_structured + ref_values

        data["model_output"].extend([item for sublist in worse_obvious for item in sublist])
        data["model_output"].extend(same_obvious)

        df = pd.DataFrame(data)

        for metric in self.metrics:
            original_values = dataset.get_metric_values(metric)
            true_outputs = dataset.get_dataframe()[dataset.get_output_column()]

            try:
                # Use predict for aggregators to avoid coefficient reconstruction and ensure exact behavior
                if isinstance(metric, Aggregator):
                    input_col = dataset.get_input_column()
                    output_col = dataset.get_output_column()
                    tmp_data = {
                        input_col: df["input"].tolist(),
                        output_col: df["model_output"].tolist(),
                    }
                    for ref_col in reference_columns:
                        tmp_data[ref_col] = df[ref_col].tolist()
                    tmp_df = pd.DataFrame(tmp_data)
                    tmp_ds = dataset.copy()
                    tmp_ds.set_dataframe(tmp_df)
                    metric.predict(tmp_ds, update_dataset=True)
                    res_series = tmp_ds.get_dataframe()[metric.get_name()].tolist()
                    results = res_series
                else:
                    results = metric.calculate_batched(
                        df["input"],
                        df["model_output"],
                        [
                            [df[ref_col].iloc[i] for ref_col in reference_columns]
                            for i in range(len(df))
                        ],
                    )
            except Exception as e:
                print(f"[Robustness] Metric '{metric.get_name() if hasattr(metric,'get_name') else type(metric).__name__}' evaluation failed: {e}")
                continue

            if isinstance(results, (list, tuple)) and isinstance(metric, MultiMetric):
                sub_names = metric.get_submetric_names()
                if not isinstance(sub_names, (list, tuple)):
                    sub_names = []
                if len(results) < len(sub_names):
                    print(f"[Robustness] MultiMetric '{metric.get_name()}' returned {len(results)} results but has {len(sub_names)} submetrics; skipping")
                    continue
                for i, submetric_name in enumerate(sub_names):
                    vals_i = list(results[i])
                    data[submetric_name] = vals_i
                    try:
                        data[submetric_name].extend(original_values[submetric_name])
                    except Exception as e:
                        print(f"[Robustness] Failed to append originals for submetric '{submetric_name}': {e}")
            else:
                mname = metric.get_name() if hasattr(metric, 'get_name') else type(metric).__name__
                data[mname] = list(results) if not isinstance(results, list) else results
                try:
                    data[mname].extend(original_values)
                except Exception as e:
                    # original_values might already be aligned; skip extend if incompatible
                    print(f"[Robustness] Failed to append originals for metric '{mname}': {e}")

        data["input"].extend(inputs)
        for ref_col in reference_columns:
            data[ref_col].extend(dataset.get_dataframe()[ref_col].tolist())
        data["model_output"].extend(true_outputs)
        data["strategy"].extend(["original"] * len(inputs))
        data["group"].extend(["original"] * len(inputs))

        return pd.DataFrame(data)

    def run(self, print_results=False, num_demonstration_examples=3, max_eval_examples=30, max_workers=8):
        test_dataset = self.test_dataset
        if max_eval_examples < len(test_dataset.get_dataframe()):
            test_dataset = test_dataset.get_subset(max_eval_examples, seed=self.seed)

        producer = ProducePerturbations(num_examples=num_demonstration_examples, max_workers=max_workers)

        if self.kwargs.get("lm"):
            self.lm = self.kwargs.get("lm")
        else:
            self.lm = dspy.settings.lm

        with dspy.settings.context(lm=self.lm):
            for column in test_dataset.get_target_columns():
                perturbations = producer.forward(
                    task=test_dataset.get_task_description(),
                    dimension=column,
                    dataset=test_dataset,
                )
                df = self._produce_perturbation_scores(test_dataset, perturbations)
                self.results[f"{column}/full_table"] = TabularResult(df)

                if print_results:
                    print(df)

                df["sample_id"] = (
                    df["input"]
                    .str.strip()
                    .str.lower()
                    .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
                )
                # Only obvious mode is used in report card; analysis may assume subtle groups.
                # Guard plotting to avoid failures when subtle groups are absent.
                try:
                    analyze_and_plot(df, self.metrics, column, self.results)
                except Exception:
                    pass

def main():
    import os
    from autometrics.dataset.datasets.simplification.simplification import SimpDA
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.SARI import SARI

    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    dataset = SimpDA()

    experiment = RobustnessExperiment(
        name="Robustness Experiment",
        description="An experiment to test the robustness of the model",
        metrics=[BLEU(), SARI()],
        output_dir="outputs/robustness",
        dataset=dataset,
    )

    experiment.run(print_results=True)
    experiment.save_results()

if __name__ == "__main__":
    main()