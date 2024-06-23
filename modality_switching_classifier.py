import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class GlobalClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = OneVsRestClassifier(LinearSVC())

    def train(self, X, y):
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)

    def predict(self, X):
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)

    def fine_tune(self, X, y):
        # Combine new data with existing data
        X_combined = list(self.classifier.classes_) + list(X)
        y_combined = list(self.classifier.classes_) + list(y)

        # Re-fit the vectorizer and classifier with the combined data
        X_tfidf = self.vectorizer.fit_transform(X_combined)
        self.classifier.fit(X_tfidf, y_combined)


class SubClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = OneVsRestClassifier(LinearSVC())

    def train(self, X, y):
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)

    def predict(self, X):
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)

    def fine_tune(self, X, y):
        # Combine new data with existing data
        X_combined = list(self.classifier.classes_) + list(X)
        y_combined = list(self.classifier.classes_) + list(y)

        # Re-fit the vectorizer and classifier with the combined data
        X_tfidf = self.vectorizer.fit_transform(X_combined)
        self.classifier.fit(X_tfidf, y_combined)


class PromptClassifier:
    def __init__(self):
        self.global_classifier = GlobalClassifier()
        self.sub_classifiers = {
            "Text_To_Image": SubClassifier(),
            "Text_To_Video": SubClassifier(),
            "Text_To_Audio": SubClassifier(),
            "Text_To_Text": SubClassifier(),
            "Image_To_Text": SubClassifier(),
            "Video_To_Text": SubClassifier(),
            "Audio_To_Text": SubClassifier(),
        }

    def train(self, data):
        # Train global classifier
        global_X, global_y = zip(
            *[
                (prompt, modality)
                for modality, tasks in data.items()
                for task, prompts in tasks.items()
                for prompt in prompts
            ]
        )
        self.global_classifier.train(global_X, global_y)

        # Train sub-classifiers
        for modality, tasks in data.items():
            sub_X, sub_y = zip(
                *[
                    (prompt, task)
                    for task, prompts in tasks.items()
                    for prompt in prompts
                ]
            )
            self.sub_classifiers[modality].train(sub_X, sub_y)

    def classify(self, prompt):
        # First level: Global classification
        modality = self.global_classifier.predict([prompt])[0]

        # Second level: Sub-classification
        task = self.sub_classifiers[modality].predict([prompt])[0]

        return modality, task

    def fine_tune(self, new_data):
        # Fine-tune global classifier
        global_X, global_y = zip(
            *[
                (prompt, modality)
                for modality, tasks in new_data.items()
                for task, prompts in tasks.items()
                for prompt in prompts
            ]
        )
        self.global_classifier.fine_tune(global_X, global_y)

        # Fine-tune sub-classifiers
        for modality, tasks in new_data.items():
            sub_X, sub_y = zip(
                *[
                    (prompt, task)
                    for task, prompts in tasks.items()
                    for prompt in prompts
                ]
            )
            self.sub_classifiers[modality].fine_tune(sub_X, sub_y)

        print("Fine-tuning completed.")

    def evaluate(self, test_data):
        true_modalities, true_tasks, prompts = [], [], []
        for modality, tasks in test_data.items():
            for task, task_prompts in tasks.items():
                true_modalities.extend([modality] * len(task_prompts))
                true_tasks.extend([task] * len(task_prompts))
                prompts.extend(task_prompts)

        predicted_modalities, predicted_tasks = zip(
            *[self.classify(prompt) for prompt in prompts]
        )

        print("Global Classifier Performance:")
        print(classification_report(true_modalities, predicted_modalities))

        print("\nSub-Classifier Performance:")
        for modality in set(true_modalities):
            modality_mask = np.array(true_modalities) == modality
            print(f"\n{modality}:")
            print(
                classification_report(
                    np.array(true_tasks)[modality_mask],
                    np.array(predicted_tasks)[modality_mask],
                )
            )


# Example usage
if __name__ == "__main__":
    # Initial training data (same as before)
    data = {
        "Text_To_Image": {
            "Image_Generation": [
                "Generate a photorealistic image of a snow-capped mountain at sunrise",
                "Create a digital painting of a bustling medieval marketplace",
                "Produce an anime-style portrait of a cyberpunk character",
                "Generate a vector illustration of a tropical bird in flight",
                "Create a surrealist image combining elements of clocks and butterflies",
                "Generate a realistic 3D render of a futuristic sports car",
                "Create a watercolor-style landscape of a serene Japanese garden",
                "Produce a pixel art scene of a space battle between alien ships",
                "Generate an abstract representation of the concept of time",
                "Create a detailed architectural blueprint of a sustainable treehouse",
            ],
            "Image_Editing": [
                "Remove the background from this product photo and replace it with a pure white backdrop",
                "Enhance the clarity and sharpness of this blurry landscape photograph",
                "Colorize this black and white historical image of New York City in the 1920s",
                "Adjust the lighting in this portrait to create a dramatic chiaroscuro effect",
                "Seamlessly remove the person standing in the foreground of this tourist photo",
                "Apply a vintage film grain effect to this digital photograph",
                "Correct the perspective distortion in this architectural photograph",
                "Retouch this fashion photo to smooth skin textures and enhance eye colors",
                "Create a tilt-shift effect on this cityscape to make it appear miniature",
                "Composite multiple exposures of a lunar eclipse into a single image",
            ],
        },
        "Text_To_Audio": {
            "Text_To_Speech": [
                "Convert this academic paper on quantum physics into spoken words with proper scientific pronunciation",
                "Read this children's bedtime story with different voices for each character",
                "Narrate this news article in the style of a professional newscaster",
                "Transform this motivational quote into an inspiring spoken affirmation",
                "Convert this technical manual into clear, articulate speech for a video tutorial",
                "Read this poem with appropriate rhythm and emotional inflection",
                "Narrate this historical document in the accent of its original time period",
                "Convert this recipe into step-by-step audio instructions for a cooking podcast",
                "Read this legal contract with clear enunciation of complex terms",
                "Transform this movie script into a dramatic audio performance with multiple voices",
            ],
            "Sound_Generation": [
                "Create a 30-second ambient sound of a peaceful forest with birds chirping and leaves rustling",
                "Generate the sound of a bustling coffee shop with background chatter and espresso machines",
                "Produce a soundscape of a thunderstorm approaching, peaking, and then receding",
                "Create the audio atmosphere of an alien planet with strange, otherworldly sounds",
                "Generate a loopable background track for meditation with gentle bells and flowing water",
                "Produce the sound of a car engine starting, idling, and then accelerating",
                "Create a realistic audio simulation of waves crashing on a rocky shore",
                "Generate the ambient sound of a busy hospital emergency room",
                "Produce a 1-minute track of futuristic computer and machinery sounds",
                "Create the audio experience of being in the middle of a cheering stadium crowd",
            ],
        },
        "Text_To_Text": {
            "Question_Answering": [
                "What are the main causes and effects of climate change?",
                "Explain the process of photosynthesis in simple terms",
                "Who were the key figures in the American Civil Rights Movement?",
                "What is the difference between machine learning and deep learning?",
                "How does the human immune system work to fight off infections?",
                "What are the primary arguments for and against universal basic income?",
                "Explain the concept of quantum entanglement in layman's terms",
                "What were the main causes of the French Revolution?",
                "How does blockchain technology work and what are its potential applications?",
                "What is the current scientific understanding of dark matter and dark energy?",
            ],
            "Summarization": [
                "Provide a concise summary of the plot of 'To Kill a Mockingbird'",
                "Summarize the key findings of the latest IPCC report on climate change",
                "Give an overview of the major events of World War II in chronological order",
                "Summarize the main principles of Maslow's hierarchy of needs",
                "Provide a brief explanation of the theory of evolution by natural selection",
                "Summarize the key points of Martin Luther King Jr.'s 'I Have a Dream' speech",
                "Give an overview of the major schools of thought in philosophy",
                "Summarize the plot and themes of George Orwell's '1984'",
                "Provide a concise explanation of how the stock market works",
                "Summarize the current understanding of the human microbiome and its importance",
            ],
        },
        "Image_To_Text": {
            "Image_Description": [
                "Describe in detail the composition and subject matter of this abstract painting",
                "What can you tell me about the architectural style and features of the building in this photograph?",
                "Analyze the body language and facial expressions of the people in this group photo",
                "Describe the natural landscape features visible in this satellite image",
                "What details can you provide about the fashion and time period depicted in this historical photograph?",
                "Describe the layout and key elements of this infographic about renewable energy",
                "What can you tell me about the species and behavior of the animals shown in this wildlife photo?",
                "Analyze the use of color, light, and shadow in this Renaissance painting",
                "Describe the key features and condition of the artifact shown in this archaeological photograph",
                "What technical details can you provide about the car model shown in this image?",
            ],
            "OCR_Text_Extraction": [
                "Extract and transcribe all text visible on the street signs and storefronts in this city photograph",
                "Read and list all the ingredients from this image of a nutrition label",
                "Transcribe the handwritten text in this image of an old letter",
                "Extract all text from this photograph of a complex scientific diagram with annotations",
                "Read and organize the text from this image of a restaurant menu",
                "Transcribe the text from this photograph of an ancient stone inscription",
                "Extract and format the text from this image of a business card",
                "Read and list all the book titles visible on the spines in this bookshelf image",
                "Transcribe the text from this image of a historical document with old-style typography",
                "Extract all visible text from this photograph of a cluttered bulletin board",
            ],
        },
        "Audio_To_Text": {
            "Sound_Identification": [
                "Identify the types of birds singing in this audio recording from a rainforest",
                "What musical instruments can you hear in this orchestral performance?",
                "Identify the make and model of the car based on the engine sound in this audio clip",
                "What types of weather phenomena can be heard in this outdoor ambient recording?",
                "Identify the genre and potential decade of origin for this music clip",
                "What types of animals can be heard in this nighttime audio recording?",
                "Identify the different kitchen appliances operating in this audio clip",
                "What types of emergency vehicles can be heard in this urban soundscape?",
                "Identify the different sports being played based on the sounds in this recording",
                "What types of power tools or machinery can be heard in this construction site audio?",
            ],
            "Speech_Transcription": [
                "Transcribe this audio recording of a fast-paced debate, identifying speakers where possible",
                "Provide a verbatim transcription of this technical lecture, including any specialized terms",
                "Transcribe this audio of a news broadcast, including any non-speech audio cues",
                "Create a time-stamped transcript of this podcast interview",
                "Transcribe this audio recording of a courtroom proceeding, noting speaker changes",
                "Provide a transcript of this multi-language business meeting, noting the language switches",
                "Transcribe this historical speech recording, noting any areas of uncertainty due to audio quality",
                "Create a transcript of this stand-up comedy routine, noting audience reactions",
                "Transcribe this audio diary entry, including any emotional cues in the speaker's voice",
                "Provide a detailed transcript of this conference call, identifying each participant",
            ],
        },
    }
    # Initialize and train the classifier
    classifier = PromptClassifier()
    classifier.train(data)

    # Test data for evaluation
    test_data = {
        "Text_To_Image": {
            "Image_Generation": [
                "Create an image of a flying dragon",
                "Generate a picture of a cyberpunk street",
            ],
            "Image_Editing": [
                "Crop this image to a square",
                "Increase the contrast of this photo",
            ],
        },
        "Text_To_Audio": {
            "Text_To_Speech": [
                "Read this news article aloud",
                "Convert this poem to speech",
            ],
            "Sound_Generation": [
                "Create the sound of a busy city",
                "Generate a bird chirping sound",
            ],
        },
        "Text_To_Text": {
            "Question_Answering": [
                "Who won the Nobel Prize in Physics in 2022?",
                "What is the boiling point of water?",
            ],
            "Summarization": [
                "Summarize the plot of Romeo and Juliet",
                "Give an overview of the French Revolution",
            ],
        },
        "Image_To_Text": {
            "Image_Description": [
                "What do you see in this landscape photo?",
                "Describe the contents of this infographic",
            ],
            "OCR_Text_Extraction": [
                "Read the text on this street sign",
                "Extract the words from this scanned document",
            ],
        },
        "Audio_To_Text": {
            "Sound_Identification": [
                "What type of animal is making this sound?",
                "Identify the genre of this music clip",
            ],
            "Speech_Transcription": [
                "Convert this podcast episode to text",
                "Transcribe this recorded lecture",
            ],
        },
    }

    # Evaluate the classifier
    print("Initial Evaluation:")
    classifier.evaluate(test_data)

    # New data for fine-tuning
    new_data = {
        "Text_To_Image": {
            "Image_Inpainting": [
                "Fill in the missing parts of this torn photograph",
                "Complete the blanked-out areas in this image",
            ],
            "Image_Outpainting": [
                "Extend this image beyond its current borders",
                "Add more scenery around the edges of this picture",
            ],
        },
        "Text_To_Audio": {
            "Music_Generation": [
                "Compose a short melody in the style of Mozart",
                "Create a hip-hop beat with a strong bassline",
            ],
        },
        "Text_To_Text": {
            "Code_Generation": [
                "Write a Python function to calculate Fibonacci numbers",
                "Create a JavaScript function for form validation",
            ],
            "Code_Debugging": [
                "Find and fix the bug in this C++ code",
                "Debug this SQL query that's producing incorrect results",
            ],
        },
        "Image_To_Text": {
            "Object_Identification": [
                "List all the objects you can see in this kitchen scene",
                "Identify the animals present in this wildlife photo",
            ],
            "Image_Palette_Extraction": [
                "What are the dominant colors in this painting?",
                "Describe the color scheme of this product packaging",
            ],
        },
        "Audio_To_Text": {
            "Music_Prompt_Generation": [
                "Describe this classical music piece in words",
                "Generate a text prompt to recreate this electronic music track",
            ],
        },
    }

    # Fine-tune the classifier
    classifier.fine_tune(new_data)

    # Re-evaluate after fine-tuning
    print("\nEvaluation after fine-tuning:")
    classifier.evaluate(test_data)

    # Test the classifier with new prompts
    test_prompts = [
        "Generate an image of a cat playing with a ball of yarn",
        "Convert this paragraph into a British accent speech",
        "What is the capital of Spain?",
        "Describe the objects in this photograph of a living room",
        "Transcribe this audio file of a stand-up comedy routine",
        "Complete the missing part of this partially damaged photo",
        "Create a jazz piano solo",
        "Debug this Python script that's throwing an IndexError",
        "What are the main colors used in this abstract painting?",
        "Generate a text description to recreate this drum beat",
    ]

    print("\nClassifying new prompts:")
    for prompt in test_prompts:
        modality, task = classifier.classify(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Modality: {modality}")
        print(f"Task: {task}\n")
