import gradio as gr
from chat_bot import ChatBot
from langchain_core.messages import HumanMessage, AIMessage


folder_path = "../data"
inference = ChatBot(folder_path)
session_id = "user_session"


def chat_with_bot(user_input):
    response = inference.generate_with_history(user_input, session_id)
    chat_history = inference.get_session_history(session_id)

    human_history = "\n".join(
        [msg.content for msg in chat_history.messages if isinstance(msg, HumanMessage)]
    )
    ai_history = "\n".join(
        [msg.content for msg in chat_history.messages if isinstance(msg, AIMessage)]
    )

    return response, human_history, ai_history

inputs = gr.Textbox(lines=2, placeholder="Enter your input here...")
outputs = [
    gr.Textbox(label="Response"),
    gr.Textbox(label="Human History", lines=10),
    gr.Textbox(label="AI History", lines=10),
]

gr.Interface(
    fn=chat_with_bot, inputs=inputs, outputs=outputs, title="Chat with AI"
).launch()
