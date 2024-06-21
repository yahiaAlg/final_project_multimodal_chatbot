from pprint import pprint
import PyPDF2
import PyPDF2.errors
import chainlit as cl
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
import logging
import os

load_dotenv(find_dotenv(), override=True)


from getpass import getpass

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Google API key: ")


@cl.on_chat_start
async def chat_message_pdf_config():
    elements = [cl.Image(name="first_image", display="inline", path="bot_avatar.jpeg")]
    await cl.Message(content="Ask any Question about the pdf", elements=elements).send()
    files = None

    while not files:
        files = await cl.AskFileMessage(
            content="place your pdfs first",
            accept=["application/pdf"],
            max_size_mb=5,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Uploading and Processing file {file.name} ...")
    await msg.send()
    pprint(file)
    with open(f"{file.path}", "rb") as filestream:
        pdf_content = PyPDF2.PdfReader(filestream)

        pdf_text = "\n".join([page.extract_text() for page in pdf_content.pages])
        print("--" * 100)
        print("pdf_text:")
        pprint(pdf_text)
        if pdf_text:

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=20
            )
            chunks = text_splitter.split_text(pdf_text)
            metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))]
            print("chunks:")
            pprint(chunks[:3])
            if chunks:
                embedding = OllamaEmbeddings(model="phi3")
                vector_store = await cl.make_async(Chroma.from_texts)(
                    chunks, embedding, metadatas=metadatas
                )

                cl.user_session.set("metadatas", metadatas)
                cl.user_session.set("chunks", chunks)
                cl.user_session.set("vector_store", vector_store)
                qachain = RetrievalQA.from_chain_type(
                    llm=Ollama(
                        model="phi3",
                    ),
                    # llm=ChatGoogleGenerativeAI(
                    #     model="gemini-pro",
                    #     temperature=0,
                    #     google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                    # ),  # type: ignore
                    verbose=True,
                    retriever=vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}
                    ),
                    chain_type="stuff",
                    return_source_documents=True,
                )
                print(" QA chain:")
                pprint(qachain)
                cl.user_session.set("qachain", qachain)
                msg.content = f"finshed processing file {file.name}"
            else:
                msg.content = f"no chunks was found!"
        else:
            msg.content = f"there is no content in {file.name}"
        await msg.update()


@cl.on_message
async def main(msg: cl.Message):
    sources = []
    qachain = cl.user_session.get("qachain")  # RetrievalQA
    lc_callback = cl.AsyncLangchainCallbackHandler(["ANSWER", "FINAL"], True)
    lc_callback.answer_reached = True
    response = await qachain.acall(msg, callbacks=[lc_callback])

    answer = response.get("answer", "")
    sources = response.get("sources", "").strip()
    sources_elements = []
    all_sources = cl.user_session.get("metadata", "")
    if sources:
        found_sources = []
        for i, source in enumerate(sources):
            current_source_meta = all_sources[i]
            found_sources.append(source.page_content[:10] + "...")
            sources_elements.append(
                cl.Text(name=current_source_meta, content=source.page_content)
            )
        if found_sources:
            answer += ",\n".join(found_sources)
        else:
            answer += "no source found!"

    if lc_callback.has_streamed_final_answer:
        lc_callback.final_stream.elements = sources_elements  # type: ignore
        await lc_callback.final_stream.update()
    else:
        await cl.Message(content=answer, elements=sources_elements).send()
