from dotenv import load_dotenv
import streamlit as st
from os import listdir, getenv
from langchain.llms import GPT4All
from langchain import ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

models_path = getenv("GPT4ALL_PATH")

if not models_path:
    st.warning(
        """Please add GPT4ALL_PATH to your env, \
that is the path from which we load the models"""
    )
    st.stop()

template = """Assistant is a large language model trained by @theopomies.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)


models = [
    model
    for model in listdir(
        models_path,
    )
    if model.endswith(".bin")
]

model = None
chatgpt_chain = None

with st.sidebar:
    st.title("Theo's Local Chatbots")

    model = st.selectbox("Select a chatbot", [None] + models)

if "chatgpt_chain" not in st.session_state:
    if model:
        print("Loading model")
        llm = GPT4All(model=models_path + model, n_threads=8)  # type: ignore

        st.session_state["chatgpt_chain"] = ConversationChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferMemory(),
        )


st.title("🦾 Theo's Chatbot")

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    chatgpt_chain = st.session_state["chatgpt_chain"]
    if not model or not chatgpt_chain:
        st.info("Please select a chatbot in the sidebar")
        st.stop()

    st.session_state["history"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    output = chatgpt_chain.predict(input=user_input)
    st.session_state["history"].append({"role": "assistant", "content": output})
    st.chat_message("assistant").write(output)
