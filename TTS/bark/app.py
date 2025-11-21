import gradio as gr
from bark_stream import stream_bark_tts

def bark_interface(text):
    for chunk in stream_bark_tts(text):
        yield chunk

demo = gr.Interface(
    fn=bark_interface,
    inputs="text",
    outputs=gr.Audio(streaming=True),
    title="🎙️ Suno Bark Streaming TTS (Test)",
    description="Generates near real-time speech using Bark. Works best with short sentences or emotion tags."
)

demo.queue()
demo.launch()
