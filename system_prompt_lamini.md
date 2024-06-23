give me system prompts for classifying

- text to text coding prompts
- text to text debugging prompts
- text to text question answer prompts
- text to text normal conversation or chatting prompts
- image to text description prompts
- image to text OCR text extraction prompts
- text to audio speech prompts
- text to audio sound prompts
- audio to text prompt describing the sound heard
- audio to text prompt writing the speech heard
- text to image , image generation prompts
- text + image ,image generation prompts from base image
- text + image ,image editing in-painting prompts
- text + image ,image editing out-painting prompts
- text + image ,image up-scaling prompts
- text + image ,image higher resolution-fix prompts

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
