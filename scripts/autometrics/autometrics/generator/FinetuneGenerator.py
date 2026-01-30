import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from sklearn.model_selection import train_test_split
import dspy
import torch
from platformdirs import user_data_dir

from autometrics.generator.Generator import Generator

# Import the metric classes (will be created next)
from autometrics.metrics.generated.GeneratedFinetunedMetric import (
    GeneratedRefFreeFinetunedMetric,
    GeneratedRefBasedFinetunedMetric
)


class FinetuneGenerator(Generator):
    """Generate fine-tuned metrics by training ModernBERT-Large models on user data.
    
    This generator fine-tunes a regression model on the provided dataset, creating
    metrics that can predict quality scores based on the learned patterns in the data.
    Unlike other generators, this creates actual learned models rather than prompting strategies.
    
    The class follows the Generator interface but has special considerations:
    - Default n_metrics=1 (fine-tuning is expensive)
    - Models are saved to user data directory
    - Supports optional HuggingFace upload
    - Uses 80/20 train/validation split
    """

    def __init__(
        self,
        name: str = "FinetuneGenerator",
        description: str = "Generate fine-tuned ModernBERT metrics based on dataset regression training",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        model_name: str = "answerdotai/ModernBERT-large",
        max_seq_length: int = 2048,
        num_train_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        upload_to_hf: bool = False,
        hf_repo_name: Optional[str] = None,
        seed: int = 42,
        model_save_dir: Optional[str] = None,
        truncate_chars: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
            truncate_chars=truncate_chars,
        )

        # Fine-tuning specific parameters
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.upload_to_hf = upload_to_hf
        self.hf_repo_name = hf_repo_name
        self.seed = seed
        self.model_save_dir_arg = model_save_dir

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

        # Set up model save directory with precedence: arg > env var > default
        if self.model_save_dir_arg:
            self.model_save_dir = Path(self.model_save_dir_arg)
            print(f"🗂️  Using model directory from argument: {self.model_save_dir}")
        elif os.environ.get("AUTOMETRICS_MODEL_DIR"):
            self.model_save_dir = Path(os.environ.get("AUTOMETRICS_MODEL_DIR"))
            print(f"🗂️  Using model directory from AUTOMETRICS_MODEL_DIR: {self.model_save_dir}")
        else:
            self.model_save_dir = Path(user_data_dir("autometrics")) / "models"
            print(f"🗂️  Using default model directory: {self.model_save_dir}")
            
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if has_references:
            return GeneratedRefBasedFinetunedMetric
        else:
            return GeneratedRefFreeFinetunedMetric

    def _prepare_training_data(self, dataset, target_measure: str, formatter: Optional[Callable] = None):
        """Prepare the dataset for training by splitting and formatting."""
        formatter = self._resolve_formatter(dataset, formatter)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        # Extract input, output, target, and references
        input_col = dataset.get_input_column()
        output_col = dataset.get_output_column()
        reference_cols = dataset.get_reference_columns()
        
        # Create text for training using the dataset's formatter
        texts = []
        for row_tuple in df.iterrows():
            formatted_text = formatter(row_tuple)
            texts.append(formatted_text)

        # Get target values and handle missing data
        targets = df[target_measure].values
        
        # Find valid (non-NaN) indices
        valid_indices = ~pd.isna(targets)
        valid_texts = [texts[i] for i in range(len(texts)) if valid_indices[i]]
        valid_targets = targets[valid_indices]
        
        print(f"📊 Data cleaning summary:")
        print(f"   Total examples: {len(texts)}")
        print(f"   Valid examples (non-NaN): {len(valid_texts)}")
        print(f"   Removed examples: {len(texts) - len(valid_texts)}")
        print(f"   Target range: [{valid_targets.min():.3f}, {valid_targets.max():.3f}]")
        
        if len(valid_texts) < 10:
            raise ValueError(f"Insufficient valid training data after removing NaN values. Only {len(valid_texts)} examples remain.")

        # 80/20 train/validation split on clean data
        train_texts, val_texts, train_targets, val_targets = train_test_split(
            valid_texts, valid_targets, test_size=0.2, random_state=self.seed, stratify=None
        )

        return train_texts, val_texts, train_targets, val_targets

    def _finetune_model(self, train_texts: List[str], train_targets: np.ndarray, 
                       val_texts: List[str], val_targets: np.ndarray, 
                       model_save_path: str) -> str:
        """Fine-tune the ModernBERT model for regression using PEFT/LoRA."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TrainingArguments, 
                Trainer,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
            from sklearn.metrics import mean_squared_error, r2_score
            import torch
        except ImportError as e:
            raise ImportError(f"Required libraries not installed: {e}. Please install transformers and peft.")

        print(f"Fine-tuning {self.model_name} for regression using PEFT/LoRA...")
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,  # Regression - single output
            torch_dtype=torch.float32,  # Use float32 for stability
        )

        # Configure LoRA for efficient fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence Classification (includes regression)
            inference_mode=False,
            r=16,  # Smaller rank for stability
            lora_alpha=32,  # Conservative scaling factor  
            lora_dropout=0.1,  # Dropout for LoRA layers
            bias="none",  # Don't train bias terms
            target_modules=["query", "value", "key", "dense"],  # Target attention and dense layers
            init_lora_weights=True,  # Ensure proper initialization
        )
        
        # Wrap base model with LoRA adapters
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()  # Show parameter efficiency
        
        # Ensure proper device placement for PEFT model
        if torch.cuda.is_available():
            model = model.cuda()

        # Prepare datasets
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=self.max_seq_length)

        # Convert targets to float and validate
        train_labels = train_targets.astype(float)
        val_labels = val_targets.astype(float)
        
        # Check for NaN or infinite values in labels
        if np.any(np.isnan(train_labels)) or np.any(np.isinf(train_labels)):
            raise ValueError(f"Training labels contain NaN or infinite values: {train_labels}")
        if np.any(np.isnan(val_labels)) or np.any(np.isinf(val_labels)):
            raise ValueError(f"Validation labels contain NaN or infinite values: {val_labels}")
            
        print(f"✅ Labels validation passed - no NaN or infinite values detected")

        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        val_dataset = Dataset.from_dict({
            'text': val_texts, 
            'labels': val_labels
        })

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Define compute metrics for regression
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # For regression, predictions is logits with shape (batch_size, 1)
            predictions = predictions.flatten()
            
            # Handle NaN predictions gracefully
            if np.any(np.isnan(predictions)):
                predictions = np.nan_to_num(predictions, nan=np.mean(labels))
            
            # Handle infinite predictions
            if np.any(np.isinf(predictions)):
                predictions = np.clip(predictions, np.min(labels) - 1, np.max(labels) + 1)
            
            # Compute regression metrics
            try:
                mse = mean_squared_error(labels, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(labels, predictions)
                
                return {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                }
            except Exception as e:
                return {
                    "mse": float('inf'),
                    "rmse": float('inf'),
                    "r2": -float('inf'),
                }

        # Set up training arguments
        training_args_config = TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=20,  # More warmup steps for stability
            fp16=False,  # Disable FP16 for stability with PEFT
            bf16=False,
            optim="adamw_torch",
            learning_rate=self.learning_rate,
            weight_decay=0.01,  # Increase weight decay for stability
            lr_scheduler_type="cosine",
            seed=self.seed,  # Use consistent seed
            num_train_epochs=self.num_train_epochs,
            save_strategy="steps",
            save_steps=0.25,
            report_to="none",
            group_by_length=True,
            eval_strategy="steps",
            eval_steps=0.25,
            logging_strategy="steps",
            logging_steps=0.25,
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False,  # Lower MSE is better
            # Early stopping based on validation performance
            save_total_limit=1,  # Keep only best 1 checkpoints
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        print("Starting training...")
        trainer_stats = trainer.train()
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Model saved to {model_save_path}")
        print(f"Training completed. Final training loss: {trainer_stats.training_loss:.4f}")

        # Upload to HuggingFace if requested
        if self.upload_to_hf and self.hf_repo_name:
            try:
                trainer.push_to_hub(self.hf_repo_name)
                print(f"Model uploaded to HuggingFace: {self.hf_repo_name}")
            except Exception as e:
                print(f"Failed to upload to HuggingFace: {e}")

        return model_save_path

    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 1, 
                formatter: Optional[Callable] = None, **kwargs) -> List:
        """
        Generate fine-tuned metrics based on the dataset.
        
        Note that n_metrics defaults to 1 for fine-tuning since it's computationally expensive.
        """
        task_description = dataset.get_task_description()

        formatter = self._resolve_formatter(dataset, formatter)
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class

        # Step-2: Prepare training data
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]
            
        print(f"Preparing training data for target measure: {target_measure}")
        train_texts, val_texts, train_targets, val_targets = self._prepare_training_data(
            dataset, target_measure, formatter
        )

        print(f"Training set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")
        print(f"Training targets - Min: {train_targets.min():.3f}, Max: {train_targets.max():.3f}, Mean: {train_targets.mean():.3f}, Std: {train_targets.std():.3f}")
        print(f"Validation targets - Min: {val_targets.min():.3f}, Max: {val_targets.max():.3f}, Mean: {val_targets.mean():.3f}, Std: {val_targets.std():.3f}")

        # Step-3: Generate metrics (typically just 1 for fine-tuning)
        new_metrics = []
        
        for i in range(n_metrics):
            # Create unique model name with seed for reproducibility
            safe_dataset_name = dataset.get_name().replace(" ", "_").replace("/", "_")
            safe_target_name = target_measure.replace(" ", "_").replace("/", "_")
            model_name = f"finetuned_{safe_dataset_name}_{safe_target_name}_seed{self.seed}_{i+1}"
            
            model_save_path = self.model_save_dir / model_name
            model_save_path.mkdir(exist_ok=True)
            
            # Step-4: Fine-tune the model
            print(f"Fine-tuning model {i+1}/{n_metrics}: {model_name}")
            final_model_path = self._finetune_model(
                train_texts, train_targets, 
                val_texts, val_targets,
                str(model_save_path)
            )

            # Step-5: Create the metric instance
            # Note: Fine-tuned metrics don't need an LLM for metric card generation
            # They generate cards programmatically using template-based approach
            
            # Validate and reconcile seed values
            executor_kwargs = self.executor_kwargs.copy()
            if self.seed is not None:
                if 'seed' in executor_kwargs and executor_kwargs['seed'] != self.seed:
                    print(f"Warning: Seed mismatch detected. Proposer seed ({self.seed}) differs from executor_kwargs seed ({executor_kwargs['seed']}). Using proposer seed.")
                executor_kwargs['seed'] = self.seed
            elif 'seed' not in executor_kwargs:
                # No seed provided anywhere, that's fine
                pass
            
            metric = dynamic_executor_class(
                name=f"{model_name}_ModernBERT",
                description=f"Fine-tuned ModernBERT metric for {target_measure} on {dataset.get_name()}",
                model_path=final_model_path,
                task_description=task_description,
                target_measure=target_measure,
                dataset_name=dataset.get_name(),
                training_stats={
                    "train_size": len(train_texts),
                    "val_size": len(val_texts),
                    "target_mean": float(np.mean(train_targets)),
                    "target_std": float(np.std(train_targets)),
                    "epochs": self.num_train_epochs,
                    "learning_rate": self.learning_rate,
                },
                metric_card_author_model=None,  # No LLM needed for programmatic generation
                **executor_kwargs,
            )

            new_metrics.append(metric)

        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 
