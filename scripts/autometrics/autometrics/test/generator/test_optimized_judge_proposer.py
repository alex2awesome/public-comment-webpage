import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import dspy
from autometrics.generator.OptimizedJudgeProposer import OptimizedJudgeProposer
from autometrics.dataset.Dataset import Dataset


class DummyModel:
    model = "test-model"
    
    def __init__(self):
        # Mimic the attribute that the real model exposes
        pass


@pytest.fixture()
def small_dataset():
    df = pd.DataFrame({
        'input': [
            "What is the capital of France?",
            "How do you make a cake?",
            "Explain quantum physics",
            "Write a poem about the sea",
            "What are the benefits of exercise?"
        ],
        'output': [
            "The capital of France is Paris.",
            "To make a cake, you need flour, eggs, and sugar...",
            "Quantum physics is the study of matter and energy...",
            "The sea waves crash upon the shore...",
            "Exercise helps improve cardiovascular health..."
        ],
        'score': [5, 3, 4, 2, 5]
    })
    return Dataset(
        dataframe=df,
        target_columns=['score'],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        input_column='input',
        output_column='output',
        task_description="Answer questions helpfully and accurately"
    )


@pytest.fixture()
def small_dataset_with_references():
    df = pd.DataFrame({
        'input': [
            "What is the capital of France?",
            "How do you make a cake?",
            "Explain quantum physics",
            "Write a poem about the sea",
            "What are the benefits of exercise?"
        ],
        'output': [
            "The capital of France is Paris.",
            "To make a cake, you need flour, eggs, and sugar...",
            "Quantum physics is the study of matter and energy...",
            "The sea waves crash upon the shore...",
            "Exercise helps improve cardiovascular health..."
        ],
        'reference': [
            "Paris is the capital and largest city of France.",
            "Cake making requires combining dry and wet ingredients...",
            "Quantum mechanics describes the behavior of subatomic particles...",
            "Ocean poetry often captures the rhythm of waves...",
            "Regular exercise benefits heart health and mental wellbeing..."
        ],
        'score': [5, 3, 4, 2, 5]
    })
    return Dataset(
        dataframe=df,
        target_columns=['score'],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset_with_refs",
        input_column='input',
        output_column='output',
        reference_columns=['reference'],
        task_description="Answer questions helpfully and accurately with reference support"
    )


@pytest.fixture()
def mock_llm_calls(monkeypatch):
    # Mock DSPy settings
    mock_settings = MagicMock()
    monkeypatch.setattr("dspy.settings", mock_settings)

    # Mock MIPROv2 optimizer
    mock_mipro = MagicMock()
    mock_optimized_program = MagicMock()
    mock_optimized_program.save = MagicMock()
    mock_mipro.compile.return_value = mock_optimized_program
    monkeypatch.setattr("autometrics.generator.OptimizedJudgeProposer.MIPROv2", lambda **kwargs: mock_mipro)
    
    # Mock chain of thought responses
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            score = "4"
            domain = "Text Generation"
            tasks = ["Task 1", "Task 2"]
            best_suited_for_circumstances = ["Circumstance 1", "Circumstance 2"]
            not_recommended_for_circumstances = ["Bad circumstance 1", "Bad circumstance 2"]
            biases = ["Bias 1", "Bias 2"]
            task_misalignment_risks = ["Risk 1", "Risk 2"]
            failure_cases = ["Failure 1", "Failure 2"]
            axes_of_variation = [
                "**Clarity**: How clear and understandable is the response?",
                "**Accuracy**: How factually correct is the information?",
                "**Completeness**: How thoroughly does the response address the question?"
            ]

        return MockOutput()

    # Mock DSPy ChainOfThought
    def mock_chain_of_thought(signature):
        class MockChainOfThought:
            def __call__(self, *args, **kwargs):
                return mock_chain_of_thought_call(self, *args, **kwargs)

        return MockChainOfThought()

    monkeypatch.setattr("dspy.ChainOfThought", mock_chain_of_thought)

    # Mock DSPy context manager
    class MockContext:
        def __init__(self, lm=None):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("dspy.settings.context", MockContext)

    # Mock generate_axes_of_variation function
    def mock_generate_axes(task_description, good_examples, bad_examples, generator_llm=None, target_name=None, num_axes_to_generate=5):
        return [
            "**Clarity**: How clear and understandable is the response?",
            "**Accuracy**: How factually correct is the information?",
            "**Completeness**: How thoroughly does the response address the question?"
        ]

    monkeypatch.setattr("autometrics.generator.OptimizedJudgeProposer.generate_axes_of_variation", mock_generate_axes)

    # Mock get_default_formatter
    def mock_get_default_formatter(dataset):
        return lambda x: str(x)

    monkeypatch.setattr("autometrics.generator.OptimizedJudgeProposer.get_default_formatter", mock_get_default_formatter)

    # Mock os.makedirs
    monkeypatch.setattr("os.makedirs", MagicMock())

    # Mock platformdirs.user_data_dir
    monkeypatch.setattr("autometrics.generator.OptimizedJudgeProposer.platformdirs.user_data_dir", lambda x: "/mock/user/data")


def test_optimized_judge_proposer_returns_expected_metrics_ref_free(mock_llm_calls, small_dataset):
    """Test that the OptimizedJudgeProposer generates the expected number of reference-free metrics."""
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_kwargs={'model': judge_model}
    )
    
    metrics = proposer.generate(small_dataset, target_measure='score', n_metrics=2)
    
    assert len(metrics) == 2
    for metric in metrics:
        assert 'optimized' in metric.name.lower()
        assert hasattr(metric, 'axis')
        assert hasattr(metric, 'optimized_prompt_path')


def test_optimized_judge_proposer_returns_expected_metrics_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that the OptimizedJudgeProposer generates the expected number of reference-based metrics."""
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_kwargs={'model': judge_model}
    )
    
    metrics = proposer.generate(small_dataset_with_references, target_measure='score', n_metrics=2)
    
    assert len(metrics) == 2
    for metric in metrics:
        assert 'optimized' in metric.name.lower()
        assert hasattr(metric, 'axis')
        assert hasattr(metric, 'optimized_prompt_path')
        assert metric.is_reference_based


def test_optimized_judge_proposer_automatic_detection(mock_llm_calls, small_dataset, small_dataset_with_references):
    """Test that the OptimizedJudgeProposer automatically detects reference vs non-reference datasets."""
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_kwargs={'model': judge_model}
    )
    
    # Test reference-free detection
    ref_free_metrics = proposer.generate(small_dataset, target_measure='score', n_metrics=1)
    assert not ref_free_metrics[0].is_reference_based
    
    # Test reference-based detection
    ref_based_metrics = proposer.generate(small_dataset_with_references, target_measure='score', n_metrics=1)
    assert ref_based_metrics[0].is_reference_based


def test_optimization_settings(mock_llm_calls, small_dataset):
    """Test that optimization settings are properly configured."""
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_kwargs={'model': judge_model},
        auto_mode="light",
        num_threads=32,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        eval_function_name='exact_match_rounded'
    )
    
    assert proposer.auto_mode == "light"
    assert proposer.num_threads == 32
    assert proposer.max_bootstrapped_demos == 4
    assert proposer.max_labeled_demos == 4


def test_custom_executor_class(mock_llm_calls, small_dataset):
    """Test using a custom executor class."""
    from autometrics.metrics.generated.GeneratedOptimizedJudge import GeneratedRefFreeOptimizedJudge
    
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_class=GeneratedRefFreeOptimizedJudge,
        executor_kwargs={'model': judge_model}
    )
    
    metrics = proposer.generate(small_dataset, target_measure='score', n_metrics=1)
    
    assert len(metrics) == 1
    assert isinstance(metrics[0], GeneratedRefFreeOptimizedJudge)


def test_generator_name_and_description():
    """Test the generator's name and description properties."""
    proposer = OptimizedJudgeProposer()
    
    assert proposer.get_name() == "OptimizedJudgeProposer"
    assert "optimized" in proposer.get_description().lower()
    assert "miprov2" in proposer.get_description().lower()


def test_metric_properties(mock_llm_calls, small_dataset):
    """Test that generated metrics have the expected properties."""
    judge_model = DummyModel()
    
    proposer = OptimizedJudgeProposer(
        generator_llm=DummyModel(),
        executor_kwargs={'model': judge_model}
    )
    
    metrics = proposer.generate(small_dataset, target_measure='score', n_metrics=1)
    metric = metrics[0]
    
    # Test basic properties
    assert hasattr(metric, 'name')
    assert hasattr(metric, 'description')
    assert hasattr(metric, 'axis')
    assert hasattr(metric, 'task_description')
    assert hasattr(metric, 'optimized_prompt_path')
    assert hasattr(metric, 'suggested_range')
    
    # Test that the suggested range matches the dataset
    expected_min = small_dataset.get_dataframe()['score'].min()
    expected_max = small_dataset.get_dataframe()['score'].max()
    assert metric.suggested_range == (expected_min, expected_max)


def test_evaluation_functions():
    """Test the evaluation functions used by MIPROv2."""
    from autometrics.generator.OptimizedJudgeProposer import exact_match_rounded, inverse_distance
    
    # Test exact_match_rounded
    assert exact_match_rounded(3.4, 3.6) == 0  # 3.4 rounds to 3, 3.6 rounds to 4
    assert exact_match_rounded(3.4, 3.4) == 1  # Both round to 3
    assert exact_match_rounded(3.0, 3.0) == 1  # Exact match
    assert exact_match_rounded(2.4, 2.6) == 0  # 2.4 rounds to 2, 2.6 rounds to 3
    
    # Test inverse_distance
    assert inverse_distance(3, 3) == 1.0  # Exact match
    assert inverse_distance(3, 4) == 1/2  # Distance 1
    assert inverse_distance(3, 5) == 1/3  # Distance 2 