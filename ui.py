import numpy as np
import gradio as gr
from lib import run

def sepia(input_img):
    output_img = run(input_img)
    return output_img

with gr.Blocks(title="yolox-test") as demo:
    gr.Markdown("# Yolox Test")
    gr.Interface(sepia, gr.Image(width=200, height=200,type = "filepath"), "image")
demo.launch(server_name="0.0.0.0", server_port=8080)
