import subprocess
import sys
import random
import time
import gradio as gr
from openai import OpenAI
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate

history = []

def clear_prompt(prompt):
    return f""


def clear_history():
    history.clear()
    return f"", f"", f"", f"", f"", f""

# def update(prompt):
#     history.append(prompt)
#     convhist = map(str,history)
#     result = '\n'.join(convhist)
#     return f"", result, do_openai(prompt), do_lmstudio(prompt), f"blue"

def do_history(prompt):
    history.append(prompt)
    convhist = map(str,history)
    return '\n'.join(convhist)


with gr.Blocks() as demo1:
    gr.Markdown("1. Select one or more models\n2. Enter system prompt\n3. prompt Model")

    with gr.Row():
        with gr.Column(scale=20):
            with gr.Row():
                models = ["llama2", "mistral", "orca-mini", "gpt-3.5-turbo", "gpt-4"]
                box1_dropdown = gr.Dropdown(choices=models,label="Select Model")
                box2_dropdown = gr.Dropdown(choices=models,label="Select Model")
                box3_dropdown = gr.Dropdown(choices=models,label="Select Model")

            
        with gr.Column(scale=1):
            sysmsg_tb = gr.Textbox(label = "System Prompt",
                            placeholder="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
                            lines=2)
            inp = gr.Textbox(label = "prompt", placeholder="type prompt")

    
    with gr.Row():
        with gr.Column(scale=20):
            with gr.Row():
                out1 = gr.Textbox(label = "model output")
                out2 = gr.Textbox(label = "model output")
                out3 = gr.Textbox(label = "model output")
        with gr.Column(scale=1):
                dummy = gr.Textbox(visible=False)
    
    def do_it(myprompt, dropdown, sysmsg):
        if sysmsg == "":
            sysmsg = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair"
        if dropdown in ("mistral", "llama2", "orca-mini"):
            start_time = time.time()
            llm = Ollama(model=dropdown)
            prompt = ChatPromptTemplate.from_messages([
                ("system", sysmsg),
                ("user", "{input}") 
            ])

            from langchain_core.output_parsers import StrOutputParser
            output_parser = StrOutputParser()

            chain = prompt | llm | output_parser
            res = chain.invoke({"input": myprompt})

            end_time = time.time()
            return res + "\n\n" + "-- elapsed time:  " + str(int(end_time - start_time)) + " seconds"
        elif dropdown in ("gpt-3.5-turbo", "gpt-4"):
            start_time = time.time()

            # Add own API key here, or set OPENAI_API_KEY environment variable
            #client = OpenAI(api_key="******")
            client = OpenAI()


            completion = client.chat.completions.create(
                model=dropdown,
                messages=[
                    #{"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "system", "content": sysmsg},
                    {"role": "user", "content": myprompt}
                ]
            )
            end_time = time.time()
            return completion.choices[0].message.content + "\n\n" + "-- elapsed time:  " + str(int(end_time - start_time)) + " seconds"


    def show_tb1(box1_dropdown):
        out1.label = box1_dropdown
        return out1

    box1_dropdown.change(show_tb1, box1_dropdown, out1)
    inp.submit(fn=do_it, inputs=[inp, box1_dropdown, sysmsg_tb], outputs=out1)
    inp.submit(fn=do_it, inputs=[inp, box2_dropdown, sysmsg_tb], outputs=out2)
    inp.submit(fn=do_it, inputs=[inp, box3_dropdown, sysmsg_tb], outputs=out3)

with gr.Blocks() as demo2:
    gr.Markdown("start demo2")


demo = gr.TabbedInterface([demo1, demo2], ["RUN", "Configuration"])
demo.queue(default_concurrency_limit=10)
#demo.launch(share=True)
demo.launch()