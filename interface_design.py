import gradio as gr
import random
import time


def bot_logic(
    message,
    chat_history,
    api_key,
    system_prompt,
    creativity,
    use_history,
    provider,
    workflow,
    task,
    style,
    negative_prompt,
    guidance,
    steps,
    seeds,
    width,
    height,
    image_format,
    save_image,
    top_k,
    chunk_size,
    chunk_overlap,
    summarization_algo,
    speech_provider,
    speech_language,
):
    # (The bot_logic function remains the same as in your original code)
    # Simulate processing time
    time.sleep(2)

    # Initialize response
    response = ""

    # Check API key
    if not api_key:
        return "", chat_history + [
            ("human", message),
            ("ai", "Please provide an API key."),
        ]

    # Apply system prompt if provided
    if system_prompt:
        response += f"[System: {system_prompt}]\n"

    # Handle different workflows
    if workflow == "Text to Text":
        if task == "conversation":
            response += (
                f"Here's a response to '{message}' with creativity {creativity}."
            )
        elif task == "translation":
            response += f"Translation: Here's the translation of '{message}'."
        elif task == "summarization":
            response += f"Summary: Here's a summary using the {summarization_algo} algorithm with chunk size {chunk_size} and overlap {chunk_overlap}."
        elif task == "question answer":
            response += f"Answer: Here's an answer to your question '{message}'."

    elif workflow == "Text to Image":
        response += f"Image generated based on '{message}' with style {style}, guidance {guidance}, steps {steps}, seeds {seeds}, dimensions {width}x{height}, format {image_format}."
        if save_image:
            response += " Image saved."

    elif workflow == "Image to Text":
        if task == "image description":
            response += f"Description: Here's a description of the uploaded image."
        elif task == "extract text from image":
            response += f"Extracted Text: Here's the text extracted from the image."
        elif task == "color pallet extraction":
            response += (
                f"Color Palette: Here are the main colors extracted from the image."
            )

    elif workflow == "Text to Audio":
        response += f"Audio generated for '{message}' using {speech_provider} in {speech_language} language."

    # Add provider information
    response += f"\n(Processed by {provider} provider)"

    # Simulate different responses based on whether to use conversation history
    if use_history:
        response += "\n(Considering conversation history)"

    return "", chat_history + [("human", message), ("ai", response)]


custom_theme = gr.themes.Default(
    spacing_size=gr.themes.sizes.spacing_lg,
    radius_size=gr.themes.sizes.radius_none,
    font=("consolas",),
    font_mono="consolas",
).set(body_text_size="2rem")

with gr.Blocks(
    theme=custom_theme,
) as demo:
    chatbot = gr.Chatbot(label="Chat Messages:")
    msg = gr.MultimodalTextbox(
        placeholder="your question:",
        label="Message:",
        lines=5,
        elem_classes="multimodal-input",
    )
    clear = gr.ClearButton([msg, chatbot])

    with gr.Accordion("Advanced:", open=False):
        api_key = gr.Textbox(label="Api key:", type="password", show_copy_button=True)
        system_prompt = gr.TextArea(
            label="System prompt:", placeholder="place your system prompt here"
        )
        creativity = gr.Number(label="Creativity", maximum=1, minimum=0, value=0.7)
        use_history = gr.Checkbox(
            label="Conversation with previous history", value=False
        )
        provider = gr.Radio(
            label="Provider", choices=["ollama", "google"], value="ollama"
        )
        workflow = gr.Dropdown(
            label="Workflows",
            choices=["Image to Text", "Text to Image", "Text to Text", "Text to Audio"],
            value="Text to Text",
        )
        task = gr.Dropdown(
            label="Workflow Tasks",
            choices=[
                "text to speech",
                "text to sound",
                "speech to text",
                "image description",
                "extract text from image",
                "color pallet extraction",
                "image creation",
                "image edition",
                "inpainting",
                "image scaling",
                "image resolution fixing",
                "programming",
                "fixing programs",
                "conversation",
                "translation",
                "summerization",
                "question answer",
            ],
            value="conversation",
        )
        style = gr.Radio(
            label="Style",
            choices=[
                "Overall",
                "Realistic",
                "Photographic",
                "Analogue",
                "Photoreal",
                "Anime",
                "Fantasy",
                "Surrealism",
                "Mix",
                "Best SDXL",
            ],
            value="Mix",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            placeholder="place your negative prompt here",
            value="easynegative,,bad face,disfigured face,bad anatomy,lowres,text,error,missing finger,cross eyed,bad proportions,gross proportions,poorly drawn face,extra foot,extra leg,disfigured,blurry,extra arm,extra hand,missing leg,missing arm,deformed limb,polar lowres,bad body,mutated hands,long neck,cropped,poorley drawn hand,bad feet",
        )
        guidance = gr.Slider(
            label="Guidance", minimum=0, maximum=1, step=0.1, value=0.7
        )
        steps = gr.Number(label="Steps", minimum=0, maximum=50, step=1, value=25)
        seeds = gr.Number(
            label="Seeds", minimum=0, maximum=2147483647, step=1, value=42
        )
        width = gr.Number(label="width", minimum=0, maximum=4096, step=1, value=512)
        height = gr.Number(label="height", minimum=0, maximum=4096, step=1, value=512)
        image_format = gr.Radio(label="Format", choices=["PNG", "JPEG"], value="JPEG")
        save_image = gr.Checkbox(label="Save Image", value=False)
        top_k = gr.Number(label="top k result:", minimum=0, maximum=20, step=1, value=4)
        chunk_size = gr.Number(
            label="chunk size", minimum=0, maximum=4096, step=1, value=1000
        )
        chunk_overlap = gr.Number(
            label="chunk overlaping", minimum=0, maximum=4096, step=1, value=200
        )
        summarization_algo = gr.Dropdown(
            label="summerization algorithm",
            choices=["stuff", "map_reduce", "refiner"],
            value="map_reduce",
        )
        speech_provider = gr.Dropdown(
            label="speech vocalist provider",
            choices=["melo", "facebook", "google"],
            value="melo",
        )
        speech_language = gr.Dropdown(
            label="speech vocalist language", choices=["ar", "en", "fr"], value="en"
        )

    msg.submit(
        bot_logic,
        [
            msg,
            chatbot,
            api_key,
            system_prompt,
            creativity,
            use_history,
            provider,
            workflow,
            task,
            style,
            negative_prompt,
            guidance,
            steps,
            seeds,
            width,
            height,
            image_format,
            save_image,
            top_k,
            chunk_size,
            chunk_overlap,
            summarization_algo,
            speech_provider,
            speech_language,
        ],
        [msg, chatbot],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
