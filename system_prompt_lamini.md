give me system prompts for classifying
-Text-To-Text workflow

- coding prompts
- debugging prompts
- question answer prompts
- creative writing prompts
- normal conversation or chatting prompts

-Image-To-Text workflow

- description prompts
- OCR text extraction prompts
- Object identification prompts
- Color Pallete extraction

-Text-To-Audio workflow

- speech prompts
- sound generation prompts
- music generation prompts

-Audio-To-Text workflow

- sound descrtiption
- speech transcription

-Text-To-Image workflow

- image generation prompts
- image generation prompts from base image
- image editing
- image up-scaling prompts
- image higher resolution-fix prompts

here is an example which make this bot create me a classifier based on criteria for the class

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts={
"cat": "Cats are generally more independent and aloof than dogs, who are often more social and affectionate. Cats are also more territorial and may be more aggressive when defending their territory.  Cats are self-grooming animals, using their tongues to keep their coats clean and healthy. Cats use body language and vocalizations, such as meowing and purring, to communicate.",
"dog": "Dogs are more pack-oriented and tend to be more loyal to their human family.  Dogs, on the other hand, often require regular grooming from their owners, including brushing and bathing. Dogs use body language and barking to convey their messages. Dogs are also more responsive to human commands and can be trained to perform a wide range of tasks.",
}

llm.prompt_train(prompts)

llm.save("models/my_model.lamini")
```

```console
llm.predict(["meow"])
>> ["cat"]

llm.predict(["meow", "woof"])
>> ["cat", "dog"]
```

now recreate the classification criteria for each of the previous given prompts types
be more detailed in describing the system prompts so the LaminiClassifier won't be making mistakes

---

Here's a more detailed classification system for the prompt types you mentioned:

Text-To-Text workflow:

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts = {
    "coding": "Prompts that request the generation, explanation, or modification of code. They often include specific programming languages, algorithms, or coding concepts. Example: 'Write a Python function to sort a list of integers.'",

    "debugging": "Prompts that ask for help identifying and fixing errors in existing code. They typically include error messages, unexpected behavior descriptions, or code snippets. Example: 'Why am I getting a 'KeyError' in this Python dictionary operation?'",

    "question_answer": "Prompts that seek factual information, explanations, or clarifications on various topics. They often start with question words (who, what, where, when, why, how) and expect concise, informative responses. Example: 'What is the capital of France?'",

    "creative_writing": "Prompts that request original, imaginative text content such as stories, poems, or scripts. They often provide a theme, character, or scenario as a starting point. Example: 'Write a short story about a time traveler who accidentally changes history.'",

    "conversation": "Prompts that simulate casual dialogue or chatting. They often include greetings, small talk, or open-ended questions about opinions or experiences. Example: 'How was your weekend? Did you do anything interesting?'"
}

llm.prompt_train(prompts)
llm.save("models/text_to_text_classifier.lamini")
```

Image-To-Text workflow:

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts = {
    "description": "Prompts that ask for a detailed verbal description of an image's content, including objects, people, actions, settings, and overall composition. Example: 'Describe everything you see in this photograph.'",

    "ocr_extraction": "Prompts that request the extraction and transcription of text visible in an image. This includes signs, labels, documents, or any other text elements. Example: 'What text can you read from the street signs in this image?'",

    "object_identification": "Prompts that ask for the identification and listing of specific objects, entities, or elements present in an image. Example: 'List all the animals you can identify in this nature photograph.'",

    "color_palette": "Prompts that request the identification and description of the main colors used in an image, often for design or artistic purposes. Example: 'What are the primary colors used in this painting?'"
}

llm.prompt_train(prompts)
llm.save("models/image_to_text_classifier.lamini")
```

Text-To-Audio workflow:

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts = {
    "speech_generation": "Prompts that request the conversion of text into spoken words, often specifying voice characteristics, emotion, or style. Example: 'Convert this paragraph into speech with a British accent.'",

    "sound_generation": "Prompts that ask for the creation of non-speech audio effects or ambient sounds based on textual descriptions. Example: 'Generate the sound of ocean waves crashing on a rocky shore.'",

    "music_generation": "Prompts that request the creation of musical compositions or melodies based on textual descriptions or parameters. Example: 'Compose a 30-second jazz tune with a prominent saxophone melody.'"
}

llm.prompt_train(prompts)
llm.save("models/text_to_audio_classifier.lamini")
```

Audio-To-Text workflow:

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts = {
    "sound_description": "Prompts that ask for a detailed verbal description of audio content, including ambient sounds, music, or other non-speech audio elements. Example: 'Describe all the sounds you can hear in this audio clip of a busy city street.'",

    "speech_transcription": "Prompts that request the conversion of spoken words in an audio file into written text. This may include specifications for handling multiple speakers, accents, or background noise. Example: 'Transcribe the conversation in this audio recording, identifying each speaker.'"
}

llm.prompt_train(prompts)
llm.save("models/audio_to_text_classifier.lamini")
```

Text-To-Image workflow:

```python
from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts = {
    "image_generation": "Prompts that request the creation of a new image based on a textual description. These often include details about style, content, composition, and mood. Example: 'Create an image of a futuristic city with flying cars and neon lights.'",

    "image_generation_from_base": "Prompts that ask for the modification or expansion of an existing image based on textual instructions. Example: 'Using this landscape as a base, add a medieval castle on the hill.'",

    "image_editing": "Prompts that request specific alterations to an existing image, such as color changes, object removal, or style transfers. Example: 'Remove the background from this product image and replace it with a plain white background.'",

    "image_upscaling": "Prompts that ask for an increase in the resolution or size of an image while maintaining or improving quality. Example: 'Upscale this low-resolution logo to 4K resolution without losing sharpness.'",

    "image_resolution_fix": "Prompts that request improvements to the clarity, sharpness, or overall quality of an image without necessarily changing its size. Example: 'Enhance the details and reduce noise in this blurry photograph.'"
}

llm.prompt_train(prompts)
llm.save("models/text_to_image_classifier.lamini")
```

These more detailed classifications should help the LaminiClassifier make more accurate distinctions between different types of prompts within each workflow category.

like you've done this one:
Certainly. For classification tasks using Hugging Face models, one of the best options is the DistilBERT model fine-tuned for sequence classification. It's a good balance of performance and efficiency. Here's how you can set up and use this model for classifying prompt types:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the labels
labels = ["Text-To-Text", "Image-To-Text", "Text-To-Audio", "Audio-To-Text", "Text-To-Image"]

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))

# Create the classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to train the model
def train_model(train_texts, train_labels):
    # Tokenize the input texts
    encodings = tokenizer(train_texts, truncation=True, padding=True)

    # Create a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(train_labels)
    )

    # Set up the trainer
    training_args = transformers.TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

# Training data
train_texts = [
    "Summarize this article in 100 words.",
    "Describe the main elements in this photograph.",
    "Convert this paragraph into spoken words.",
    "Transcribe the speech in this audio file.",
    "Create an image of a sunset over a mountain range."
]
train_labels = [0, 1, 2, 3, 4]  # Corresponding to the order in 'labels'

# Train the model
train_model(train_texts, train_labels)

# Function to classify prompts
def classify_prompt(prompt):
    result = classifier(prompt)[0]
    label_id = int(result['label'].split('_')[-1])
    return labels[label_id]

# Example usage
prompts_to_classify = [
    "Translate this sentence to French.",
    "What objects can you see in this image?",
    "Convert this text to speech with a British accent.",
    "Transcribe the conversation in this audio file.",
    "Generate an image of a futuristic cityscape."
]

for prompt in prompts_to_classify:
    print(f"Prompt: {prompt}")
    print(f"Classification: {classify_prompt(prompt)}\n")
```

This script does the following:

1. It uses the DistilBERT model pre-trained for sequence classification.
2. We define our custom labels for the prompt types.
3. The `train_model` function is used to fine-tune the model on our specific classification task.
4. We provide a small set of example prompts and their corresponding labels for training.
5. After training, we use the `classify_prompt` function to classify new prompts.

Note that this is a simplified example. For best results, you would need:

1. A larger and more diverse training dataset.
2. Proper splitting of your data into training and validation sets.
3. Hyperparameter tuning to optimize the model's performance.
4. Evaluation metrics to assess the model's accuracy.

Also, remember that the performance of this classifier heavily depends on the quality and quantity of the training data. The more diverse and numerous the examples in your training set, the better the model will perform on new, unseen prompts..

[
{"prompt":<prompt>,"classification":<classification>}, # 01th prompt
....
{"prompt":<prompt>,"classification":<classification>}, # nth prompt
]
