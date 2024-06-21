# system and utility imports
import os
import logging
import traceback
from typing import List

# gradio interface library imports
import gradio as gr

# langchain imports
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_community.llms.ollama import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Configure logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


css = """
.full_height {
    height:100vh;
} 
"""


# User message addition logic
def user(message: str, history: List[List]):
    logger.info("User function called with message: %s", message)
    try:
        return message, history + [[message, None]]
    except Exception as e:
        logger.error("Error in user function: %s", e)
        logger.debug(traceback.format_exc())
        raise


# Bot response logic
def bot(
    history: List[List[str]],
    model: str,
    temperature: float,
    api_key: str = "",
):
    logger.info(
        "Bot function called with model: %s, temperature: %f", model, temperature
    )
    try:
        history[-1][-1] = ""  # type: ignore
        match model:
            case "gemini-pro":
                if "GOOGLE_API_KEY" in os.environ:
                    logger.info("instantiating ChatGoogleGenerativeAI")
                    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)  # type: ignore
                else:
                    logger.info("instantiating ChatGoogleGenerativeAI")
                    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)  # type: ignore
            case "phi3":
                logger.info("instantiating Ollama")
                llm = Ollama(model=model)
            case _:
                raise ValueError(f"Unsupported model: {model}")

        # if model == "gemini-pro":
        #     logger.info("Using gemini-pro model")
        #     for chunk in llm.stream(history[-1][0]):
        #         logger.info("Received chunk: %s", chunk)
        #         history[-1][1] += chunk.content  # Ensure content is properly added
        #         yield history
        # elif model == "phi3":
        #     logger.info("Using phi3 model")
        #     pprint(llm.base_url)
        #     for chunk in llm.stream(history[-1][0]):
        #         logger.info("Received chunk: %s", chunk)
        #         history[-1][1] += chunk
        #         yield history
        # else:
        #     logger.error("Not a supported model: %s", model)
        #     yield "not a supported model!"
        if model == "gemini-pro":
            logger.info("Using gemini-pro model")
            response = llm.invoke(history[-1][0])
            logger.info("Received response: %s", response)
            history[-1][1] += response.content  # type: ignore # Ensure content is properly added
            return history
        elif model == "phi3":
            logger.info("Using phi3 model")
            response = llm.invoke(history[-1][0])
            logger.info("Received response: %s", response)
            history[-1][1] += response  # type: ignore # Ensure content is properly added
            return history
        else:
            logger.error("Not a supported model: %s", model)
            return "not a supported model!"
    except Exception as e:
        logger.error("Error in bot function: %s", e)
        logger.debug(traceback.format_exc())
        raise


# Interface design
def main():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="teal",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=css,
    ) as main_app:
        gr.Markdown("# Multimodal LLM")
        with gr.Row():
            ## inputs
            with gr.Column():
                with gr.Tab(label="Text Workflows"):
                    message = gr.Textbox(
                        label="User Prompt",
                        placeholder="Place your prompt here",
                    )
                with gr.Accordion(label="Advanced Configs"):
                    temperature = gr.Slider(label="temperature", maximum=1, step=0.1)
                    model = gr.Dropdown(
                        choices=["gemini-pro", "phi3"],
                        value="gemini-pro",
                        label="Models",
                    )
                    system_prompt = gr.TextArea(
                        label="System Prompt:",
                        placeholder="place your System Prompt here",
                    )

                @gr.render(inputs=model)
                def display_api_key_input(chosen_model: str):
                    try:
                        if (
                            "gemini" in chosen_model.lower()
                            and "GOOGLE_API_KEY" not in os.environ
                        ):
                            api_key = gr.Textbox(
                                "",
                                label="Google API Key",
                                placeholder="Place your API key here",
                                type="password",
                            )

                    except Exception as e:
                        logger.error("Error in display_api_key_input function: %s", e)
                        logger.debug(traceback.format_exc())
                        raise

            ## bot output
            with gr.Column(elem_classes=["full_height"]):
                chatbot = gr.Chatbot(label="Bot Response", height="100%")
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            clear_btn = gr.ClearButton(variant="stop")
        with gr.Row():
            examples = gr.Examples(
                [
                    "write me the code for bubble sort in javascript",
                    "write me the code for djikstra algorithm",
                ],
                message,
            )
        gr.on(
            [message.submit, generate_btn.click],
            user,
            [message, chatbot],
            [message, chatbot],
        ).then(
            bot,
            [chatbot, model, temperature],
            [chatbot],
            queue=False,
        )
        clear_btn.click(lambda: ("", None), None, [message, chatbot], queue=False)
    # Driver code
    try:
        main_app.queue()
        main_app.launch(debug=True)
    except Exception as e:
        logger.error("Error in main function: %s", e)
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
