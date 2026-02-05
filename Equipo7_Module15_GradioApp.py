"""
Trustworthy AI Explainer - Gradio Prototype
Module 15 Team Project Template
"""

import gradio as gr
import os
import json
import time
import zipfile
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXPORT_DIR = "exports"
CHAT_FILE = os.path.join(EXPORT_DIR, "chat_history.json")
EXPLANATION_FILE = os.path.join(EXPORT_DIR, "explanations.json")
FEEDBACK_FILE = os.path.join(EXPORT_DIR, "feedback.json")
ZIP_FILE = os.path.join(EXPORT_DIR, "trustworthy_ai_results.zip")

os.makedirs(EXPORT_DIR, exist_ok=True)

# ============================================================================
# HELPERS
# ============================================================================

def append_json(path, data):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    else:
        content = []

    content.append(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

def export_results():
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in [CHAT_FILE, EXPLANATION_FILE, FEEDBACK_FILE]:
            if os.path.exists(file):
                zipf.write(file, arcname=os.path.basename(file))
    return ZIP_FILE

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_response(
    message: str,
    history: list,
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def compute_explanation(user_text: str, response: str) -> str:
    return f"""
Explainability Analysis

Input length: {len(user_text.split())} words
Response length: {len(response.split())} words

Key factors:
- System prompt
- Temperature
- Conversation history

Placeholder for SHAP or LIME
"""


def chatbot_with_explanation(
    message: str,
    history: list,
    temperature: float,
    max_tokens: int,
    system_prompt: str
):
    response = generate_response(
        message,
        history,
        temperature,
        max_tokens,
        system_prompt
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    explanation = compute_explanation(message, response)

    append_json(CHAT_FILE, {
        "timestamp": time.time(),
        "user": message,
        "assistant": response
    })

    append_json(EXPLANATION_FILE, {
        "timestamp": time.time(),
        "explanation": explanation
    })

    return history, explanation


def handle_feedback(
    rating: str,
    comment: str
) -> str:
    if not rating:
        return "Please select a rating first."

    append_json(FEEDBACK_FILE, {
        "timestamp": time.time(),
        "rating": rating,
        "comment": comment
    })

    return "Feedback received. Thank you."

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app() -> gr.Blocks:

    with gr.Blocks(title="Trustworthy AI Explainer") as app:
        gr.Markdown("# Trustworthy AI Explainer")
        gr.Markdown("Interactive chatbot with explainability and feedback")

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400
                    )

                    message_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your question here",
                        lines=2
                    )

                    with gr.Row():
                        send_btn = gr.Button("Send")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    gr.Markdown("Explainability")
                    explanation_output = gr.Markdown(
                        value="Explanation will appear here."
                    )

            with gr.Accordion("Advanced Settings", open=False):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant.",
                    lines=2
                )

                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1)
                max_tokens = gr.Slider(50, 2000, value=500, step=50)

            gr.Markdown("Feedback")

            feedback_rating = gr.Radio(
                choices=["Thumbs Up", "Thumbs Down"],
                label="Rate the response"
            )

            feedback_comment = gr.Textbox(
                label="Optional comment",
                lines=2
            )

            feedback_btn = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(interactive=False)

            export_btn = gr.Button("Export Results")
            export_file = gr.File(label="Download ZIP")

            def submit_message(message, history, temp, tokens, sys_prompt):
                if not message.strip():
                    return history, ""

                history, explanation = chatbot_with_explanation(
                    message,
                    history,
                    temp,
                    tokens,
                    sys_prompt
                )

                return history, explanation

            send_btn.click(
                fn=submit_message,
                inputs=[message_input, chatbot, temperature, max_tokens, system_prompt],
                outputs=[chatbot, explanation_output]
            ).then(
                fn=lambda: "",
                outputs=message_input
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=chatbot
            )

            feedback_btn.click(
                fn=handle_feedback,
                inputs=[feedback_rating, feedback_comment],
                outputs=feedback_output
            )

            export_btn.click(
                fn=export_results,
                outputs=export_file
            )

        with gr.Tab("About"):
            gr.Markdown("""...""")

        with gr.Tab("Documentation"):
            gr.Markdown("""...""")

    return app

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
