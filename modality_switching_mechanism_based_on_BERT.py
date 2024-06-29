import logging
import json
import torch
import optuna
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_metric
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
import nlpaug.augmenter.word as naw
import shutil


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_threshold=0.9):
        self.early_stopping_threshold = early_stopping_threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_f1", 0) > self.early_stopping_threshold:
            control.should_training_stop = True
            logger.info(
                f"F1 score {metrics['eval_f1']:.4f} exceeded threshold {self.early_stopping_threshold}. Stopping training."
            )
        return control


class BestModelCallback(TrainerCallback):
    def __init__(self, save_threshold=0.0):
        self.best_metric = 0
        self.save_threshold = save_threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get("eval_f1", 0)
        if metric_value > self.best_metric and metric_value >= self.save_threshold:
            self.best_metric = metric_value
            # Save metrics and hyperparameters
            output_dir = f"./best_model_info"
            os.makedirs(output_dir, exist_ok=True)

            # Save metrics
            with open(f"{output_dir}/best_metrics.json", "w") as f:
                json.dump(metrics, f)

            # Save hyperparameters
            with open(f"{output_dir}/hyperparameters.json", "w") as f:
                json.dump(args.to_dict(), f)

            logger.info(
                f"New best model metrics saved with F1 score: {metric_value:.3f}"
            )


# Additional helper functions
def check_disk_space(path, required_space_gb):
    total, used, free = shutil.disk_usage(path)
    free_space_gb = free // (2**30)  # Convert bytes to GB
    if free_space_gb < required_space_gb:
        raise RuntimeError(
            f"Not enough disk space. {free_space_gb}GB available, {required_space_gb}GB required."
        )


def load_data(file_path):
    # Define the complete list of expected labels
    labels = [
        "Text-To-Text",
        "Image-To-Text",
        "Text-To-Image",
        "Text-To-Audio",
        "Audio-To-Text",
    ]

    with open(file_path, "r") as f:
        data = json.load(f)

    texts = []
    label_indices = []
    unknown_labels = set()

    for item in data:
        texts.append(item["prompt"])
        try:
            label_indices.append(labels.index(item["classification"]))
        except ValueError:
            unknown_labels.add(item["classification"])
            # Assign a default label index (e.g., -1) for unknown labels
            label_indices.append(-1)

    if unknown_labels:
        logger.warning(f"Unknown labels found in the dataset: {unknown_labels}")
        logger.warning(
            "These will be assigned a label index of -1. You may want to update your label list."
        )

    return texts, label_indices


class CustomDataset(Dataset):
    # def __init__(self, texts, labels, tokenizer, max_length=512):
    def __init__(self, texts, labels, tokenizer, max_length=128):  # Reduced from 512
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"Total GPU memory: {gpu_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be done on CPU.")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=1
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Define the PerformanceCallback class
class PerformanceCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.best_f1 = 0
        self.stagnant_epochs = 0
        self.patience = patience

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_f1 = metrics.get("eval_f1", 0)
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.stagnant_epochs = 0
            logger.info(f"New best F1 score: {self.best_f1}")
        else:
            self.stagnant_epochs += 1
            if self.stagnant_epochs >= self.patience:
                logger.warning(
                    f"Model performance stagnant for {self.stagnant_epochs} epochs. Consider early stopping or adjusting hyperparameters."
                )


# Function to save training logs
def save_training_logs(trainer, filename="training_logs.txt"):
    with open(filename, "w") as f:
        for log in trainer.state.log_history:
            f.write(json.dumps(log) + "\n")
    logger.info(f"Training logs saved to {filename}")


# metrics function
def hyperparameter_tuning(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_name,
    num_trials=10,
    timeout=3600,
):
    def objective(trial):
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        # batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        num_epochs = trial.suggest_int("num_epochs", 3, 20)

        # Address class imbalance
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(set(train_labels))
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
        val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_results = trainer.evaluate()
        return eval_results["eval_f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials, timeout=timeout)

    best_params = study.best_params
    best_f1 = study.best_value

    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best F1 score: {best_f1}")

    return best_params


def assess_dataset_quality(texts, labels, min_samples_per_class=100):
    class_counts = Counter(labels)
    total_samples = len(texts)
    num_classes = len(set(labels))

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class distribution: {class_counts}")

    if total_samples < num_classes * min_samples_per_class:
        logger.warning("Dataset might be too small.")
        return False

    if min(class_counts.values()) < min_samples_per_class:
        logger.warning("Some classes have too few samples.")
        return False

    if max(class_counts.values()) / min(class_counts.values()) > 10:
        logger.warning("Dataset is highly imbalanced.")
        return False

    logger.info("Dataset appears to be of good quality.")
    return True


# def train_model_with_type(train_texts, train_labels, val_texts, val_labels, model_type='distilbert'):
#     model_name_map = {
#       'bert': 'prajjwal1/bert-tiny',  # A much smaller BERT model
#       'roberta': 'roberta-base',
#       'distilbert': 'distilbert-base-uncased'
#     }

#     model_name = model_name_map.get(model_type.lower())
#     if model_name is None:
#         raise ValueError(f"Unknown model type: {model_type}")

#     best_params = hyperparameter_tuning(train_texts, train_labels, val_texts, val_labels, model_name)

#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
#     val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

#     training_args = TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=best_params['num_epochs'],
#         per_device_train_batch_size=best_params['batch_size'],
#         learning_rate=best_params['learning_rate'],
#         per_device_eval_batch_size=64,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         save_strategy="no",  # Don't save checkpoints
#         load_best_model_at_end=False,  # We're not saving checkpoints, so we can't load them
#         metric_for_best_model='f1',
#         greater_is_better=True,
#         push_to_hub=False,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics,
#         callbacks=[BestModelCallback(save_threshold=0.0)]  # Set threshold to 0 to always save the best model
#     )

#     performance_callback = PerformanceCallback()
#     trainer.add_callback(performance_callback)
#     trainer.train()

#     eval_results = trainer.evaluate()
#     logger.info(f"Final evaluation results: {eval_results}")

#     # Save the final model
#     final_output_dir = f'./final_model_{model_type}'
#     model.save_pretrained(final_output_dir)
#     tokenizer.save_pretrained(final_output_dir)
#     logger.info(f"Final model saved to {final_output_dir}")

#     return model, tokenizer, eval_results, trainer


def train_model_with_type(
    train_texts, train_labels, val_texts, val_labels, model_type="distilbert"
):
    model_name_map = {
        "bert": "prajjwal1/bert-tiny",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
    }

    model_name = model_name_map.get(model_type.lower())
    if model_name is None:
        raise ValueError(f"Unknown model type: {model_type}")

    best_params = hyperparameter_tuning(
        train_texts, train_labels, val_texts, val_labels, model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(set(train_labels))
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=best_params["num_epochs"],
        per_device_train_batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_threshold=0.9)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, BestModelCallback(save_threshold=0.9)],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")

    # Save the final model
    final_output_dir = f"./final_model_{model_type}"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"Final model saved to {final_output_dir}")

    return model, tokenizer, eval_results, trainer


def train_single_model(train_texts, train_labels, val_texts, val_labels, model_type):
    logger.info(f"Training {model_type} model...")
    model, tokenizer, eval_results, trainer = train_model_with_type(
        train_texts, train_labels, val_texts, val_labels, model_type
    )
    logger.info(
        f"{model_type} model training completed. Evaluation results: {eval_results}"
    )

    # Load the best model
    best_model_path = "./best_model"
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

    return eval_results, best_model, tokenizer


def augment_data(texts, labels):
    augmenter = naw.SynonymAug(aug_src="wordnet")
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        augmented_text = augmenter.augment(text)
        augmented_texts.append(augmented_text)
        augmented_labels.append(label)

    return texts + augmented_texts, labels + augmented_labels


def save_model(model, tokenizer, output_dir):
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
    except RuntimeError as e:
        logger.error(f"Failed to save model due to: {str(e)}")
        logger.info("Attempting to save model in a different format...")
        torch.save(model.state_dict(), f"{output_dir}/model_state_dict.pt")
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model state dict and tokenizer saved to {output_dir}")


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info(f"Model and tokenizer loaded from {model_dir}")
    return model, tokenizer


def classify_prompt(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class index back to the label
    labels = [
        "Text-To-Text",
        "Image-To-Text",
        "Text-To-Image",
        "Text-To-Audio",
        "Audio-To-Text",
    ]
    return labels[predicted_class]


if __name__ == "__main__":
    # Load your data
    texts, labels = load_data("prompt_dataset.json")

    # Assess dataset quality
    if not assess_dataset_quality(texts, labels):
        logger.warning("Consider improving the dataset before proceeding.")

    # Augment data if needed
    texts, labels = augment_data(texts, labels)

    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Check GPU memory
    check_gpu_memory()

    # Train models sequentially
    model_types = ["bert", "roberta", "distilbert"]
    results = {}
    best_models = {}

    # for model_type in model_types:
    #     try:
    #         eval_results, best_model, tokenizer = train_single_model(train_texts, train_labels, val_texts, val_labels, model_type)
    #         results[model_type] = eval_results
    #         best_models[model_type] = (best_model, tokenizer)
    #         save_training_logs(trainer, f'training_logs_{model_type}.txt')
    #     except Exception as e:
    #         logger.error(f"Error training {model_type} model: {str(e)}")

    # # Compare results
    # for model_type, eval_results in results.items():
    #     logger.info(f"{model_type} model results: {eval_results}")

    for model_type in model_types:
        try:
            eval_results, best_model, tokenizer = train_single_model(
                train_texts, train_labels, val_texts, val_labels, model_type
            )
            results[model_type] = eval_results
            best_models[model_type] = (best_model, tokenizer)
            save_training_logs(trainer, f"training_logs_{model_type}.txt")

            # Stop if F1 score is above 90%
            if eval_results["eval_f1"] > 0.9:
                logger.info(
                    f"F1 score {eval_results['eval_f1']:.4f} exceeded 90%. Stopping further training."
                )
                break
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")

    if results:
        # Find the best model
        best_model_type = max(results, key=lambda k: results[k]["eval_f1"])
        logger.info(f"Best model: {best_model_type}")

        # Use the best model for classification
        model, tokenizer = best_models[best_model_type]

        # Example usage of classification
        prompts_to_classify = [
            "Translate this sentence to French.",
            "What objects can you see in this image?",
            "Convert this text to speech with a British accent.",
            "Transcribe the conversation in this audio file.",
            "Generate an image of a futuristic cityscape.",
        ]

        for prompt in prompts_to_classify:
            logger.info(f"Prompt: {prompt}")
            logger.info(
                f"Classification: {classify_prompt(prompt, model, tokenizer)}\n"
            )
    else:
        logger.error(
            "No models were successfully trained. Please check the errors and try again."
        )
