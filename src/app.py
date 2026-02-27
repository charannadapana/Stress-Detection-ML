import gradio as gr

def launch_app(best_model, tfidf):

    def predict(text):
        vec = tfidf.transform([text])
        pred = best_model.predict(vec)[0]
        return "🔴 STRESSED (1)" if pred == 1 else "🟢 NOT STRESSED (0)"

    ui = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=2),
        outputs="text",
        title="Mental Health Stress Detection"
    )

    ui.launch()