import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def load_dataset(json_data):
    dataset = json.loads(json_data)
    train_data = []
    for item in dataset:
        class_name = item["class_name"]
        for example in item["examples"]:
            train_data.append({"text": example, "label": class_name})
    return train_data


def prepare_data(train_data):
    train_texts = [example["text"] for example in train_data]
    train_labels = [example["label"] for example in train_data]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2
    )

    label2id = {label: idx for idx, label in enumerate(set(train_labels))}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels = [label2id[label] for label in train_labels]
    val_labels = [label2id[label] for label in val_labels]

    return train_texts, val_texts, train_labels, val_labels, label2id, id2label


def tokenize_data(tokenizer, train_texts, val_texts):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    return train_encodings, val_encodings


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_datasets(train_encodings, val_encodings, train_labels, val_labels):
    train_dataset = PromptDataset(train_encodings, train_labels)
    val_dataset = PromptDataset(val_encodings, val_labels)
    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return model, trainer


def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")
    if "eval_accuracy" in eval_results:
        print(f"Validation Accuracy: {eval_results['eval_accuracy']}")
    else:
        print("Validation Accuracy metric not found in evaluation results.")


def classify_prompt(model, tokenizer, id2label, prompt):
    encodings = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**encodings)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return id2label[predicted_class_id]


def main():
    json_data = """
[
    {"class_name":"text_to_text_coding", "examples":["Write a Python function to reverse a string. Another", "Implement the quicksort algorithm in Java."]},
    {"class_name":"text_to_text_debugging", "examples":["Find the bug in this JavaScript code: `function add(a, b) { return a - b; }`. Another", "Debug this Python code to fix the IndexError."]},
    {"class_name":"text_to_text_qa", "examples":["What is the capital of France? Another", "Explain the theory of relativity."]},
    {"class_name":"text_to_text_chat", "examples":["How was your day? Another", "What are your plans for the weekend?"]},
    {"class_name":"image_to_text_description", "examples":["Describe the content of this image, including any objects and activities shown. Another", "What is happening in this photo?"]},
    {"class_name":"image_to_text_ocr", "examples":["Extract the text from this image. Another", "Identify and transcribe the text on this sign."]},
    {"class_name":"text_to_audio_speech", "examples":["Convert this text to speech: 'Hello, how are you?'. Another", "Generate a speech audio for this paragraph about climate change."]},
    {"class_name":"text_to_audio_sound", "examples":["Generate a sound of a cat meowing. Another", "Create a sound effect of thunder."]},
    {"class_name":"audio_to_text_sound_description", "examples":["Listen to this audio and describe the sound. Another", "What animal sounds can you hear in this clip?"]},
    {"class_name":"audio_to_text_speech_transcription", "examples":["Transcribe this speech audio to text. Another", "Write down the dialogue from this audio recording."]},
    {"class_name":"text_to_image_generation", "examples":["Generate an image of a sunset over the mountains. Another", "Create an illustration of a futuristic city."]},
    {"class_name":"text_image_to_image_generation", "examples":["Generate an image based on this base image with additional elements. Another", "Add a rainbow to this landscape image."]},
    {"class_name":"text_image_to_image_inpainting", "examples":["Edit this image to remove the object in the center. Another", "Fill in the missing parts of this damaged photo."]},
    {"class_name":"text_image_to_image_outpainting", "examples":["Extend the background of this image. Another", "Add more scenery to the edges of this photo."]},
    {"class_name":"text_image_to_image_upscaling", "examples":["Increase the resolution of this image. Another", "Upscale this low-resolution photo to make it clearer."]},
    {"class_name":"text_image_to_image_resolution_fix", "examples":["Improve the resolution quality of this image. Another", "Fix the resolution of this blurry picture."]}
]
"""

    # Load dataset
    train_data = load_dataset(json_data)

    # Prepare data
    train_texts, val_texts, train_labels, val_labels, label2id, id2label = prepare_data(
        train_data
    )

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize data
    train_encodings, val_encodings = tokenize_data(tokenizer, train_texts, val_texts)

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        train_encodings, val_encodings, train_labels, val_labels
    )

    # Train model
    model, trainer = train_model(train_dataset, val_dataset)

    # Evaluate model
    evaluate_model(trainer)

    # Classify a new prompt
    test_prompt = "Give a brief description of this image."
    predicted_class = classify_prompt(model, tokenizer, id2label, test_prompt)
    print(f"The predicted class for the prompt '{test_prompt}' is: {predicted_class}")


if __name__ == "__main__":
    main()
