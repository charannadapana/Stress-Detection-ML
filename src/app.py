import gradio as gr
import numpy as np

def launch_app(best_model, tfidf):

    def predict(text):
        if not text.strip():
            return "⚠️ Please enter some text.", 0
        
        vec = tfidf.transform([text])
        pred = best_model.predict(vec)[0]
        
        # If model supports probability (like LogisticRegression, NaiveBayes)
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(vec)[0][1]
        else:
            prob = 0.5  # fallback if no probability available
        
        if pred == 1:
            result = f"""
            🔴 **STRESSED**
            
            Stress Probability: {prob*100:.2f}%
            
            💬 Try relaxation, talk to someone, or take a short break.
            """
        else:
            result = f"""
            🟢 **NOT STRESSED**
            
            Stress Probability: {prob*100:.2f}%
            
            😊 You seem to be doing well!
            """

        return result, prob*100


    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        
        gr.Markdown(
            """
            # 🧠 Mental Health Stress Detection
            Enter a sentence and the model will predict whether it indicates stress.
            """
        )

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter your thoughts here",
                    placeholder="Example: I feel overwhelmed with my assignments...",
                    lines=4
                )
                
                predict_btn = gr.Button("🔍 Analyze Stress")
                clear_btn = gr.Button("🗑 Clear")

            with gr.Column():
                output_text = gr.Markdown()
                probability_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label="Stress Probability (%)",
                    interactive=False
                )

        # Example inputs
        gr.Examples(
            examples=[
                ["I am feeling very anxious about my exams."],
                ["Today was a great day, I am happy."],
                ["I cannot handle this workload anymore."],
                ["I enjoyed spending time with my friends."]
            ],
            inputs=text_input
        )

        # Button actions
        predict_btn.click(predict, inputs=text_input, outputs=[output_text, probability_bar])
        clear_btn.click(lambda: ("", 0), outputs=[output_text, probability_bar])

        # Optional: Live prediction while typing
        text_input.submit(predict, inputs=text_input, outputs=[output_text, probability_bar])

import os

port = int(os.environ.get("PORT", 7860))

ui.launch(
    server_name="0.0.0.0",
    server_port=port
)