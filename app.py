import gradio as gr
from text_to_video import model_t2v_fun,setup_seed
from omegaconf import OmegaConf
import torch
import imageio
import os
import cv2
import pandas as pd
import torchvision
import random
from models import get_models

from pipelines.pipeline_videogen import VideoGenPipeline
from download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


config_path = "./base/configs/sample.yaml"
args = OmegaConf.load("./base/configs/sample.yaml")
#device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif torch.backends.mps.is_available():
  device = "mps"
else:
  device = "cpu"

css = """
h1 {
  text-align: center;
}
#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

dtype = torch.float16
if device == "mps":
  dtype = torch.float32

sd_path = args.pretrained_path
unet = get_models(args, sd_path).to(device, dtype=dtype)
state_dict = find_model("./pretrained_models/lavie_base.pt")
unet.load_state_dict(state_dict)
vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=dtype).to(device)
tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=dtype).to(device) # huge
unet.eval()
vae.eval()
text_encoder_one.eval()

def infer(prompt, seed_inp, ddim_steps,cfg, infer_type):
    if seed_inp!=-1:
        setup_seed(seed_inp)
    else:
        seed_inp = random.choice(range(10000000))
        setup_seed(seed_inp)
    if infer_type == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path, 
											   subfolder="scheduler",
											   beta_start=args.beta_start, 
											   beta_end=args.beta_end, 
											   beta_schedule=args.beta_schedule)
    elif infer_type == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
        									   subfolder="scheduler",
											   beta_start=args.beta_start,
											   beta_end=args.beta_end,
											   beta_schedule=args.beta_schedule)
    elif infer_type == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
											  subfolder="scheduler",
											  beta_start=args.beta_start,
											  beta_end=args.beta_end,
											  beta_schedule=args.beta_schedule)
    model = VideoGenPipeline(vae=vae, text_encoder=text_encoder_one, tokenizer=tokenizer_one, scheduler=scheduler, unet=unet)
    model.to(device)
    if device == "cuda":
        model.enable_xformers_memory_efficient_attention()
    videos = model(prompt, video_length=16, height = 320, width= 512, num_inference_steps=ddim_steps, guidance_scale=cfg).video
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    torchvision.io.write_video(args.output_folder + prompt[0:30].replace(' ', '_') + '-'+str(seed_inp)+'-'+str(ddim_steps)+'-'+str(cfg)+ '-.mp4', videos[0], fps=8)


    return args.output_folder + prompt[0:30].replace(' ', '_') + '-'+str(seed_inp)+'-'+str(ddim_steps)+'-'+str(cfg)+ '-.mp4'


title = """
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Intern·Vchitect (Text-to-Video)
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Apply Intern·Vchitect to generate a video 
        </p>
    </div>
"""

with gr.Blocks(css='style.css') as demo:
    gr.Markdown("<font color=red size=10><center>LaVie: Text-to-Video generation</center></font>")
    gr.Markdown(
        """<div style="text-align:center">
        [<a href="https://arxiv.org/abs/2309.15103">Arxiv Report</a>] | [<a href="https://vchitect.github.io/LaVie-project/">Project Page</a>] | [<a href="https://github.com/Vchitect/LaVie">Github</a>]</div>
        """
    )
    with gr.Column():
        with gr.Row(elem_id="col-container"):
            with gr.Column():
                    
                prompt = gr.Textbox(value="a corgi walking in the park at sunrise, oil painting style", label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in", min_width=200, lines=2)
                infer_type = gr.Dropdown(['ddpm','ddim','eulerdiscrete'], label='infer_type',value='ddim')
                ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=50, step=1)
                seed_inp = gr.Slider(value=-1,label="seed (for random generation, use -1)",show_label=True,minimum=-1,maximum=2147483647)
                cfg = gr.Number(label="guidance_scale",value=7.5)

            with gr.Column():
                submit_btn = gr.Button("Generate video")
                video_out = gr.Video(label="Video result", elem_id="video-output")

            inputs = [prompt, seed_inp, ddim_steps, cfg, infer_type]
            outputs = [video_out]

        ex = gr.Examples(
            examples = [['a corgi walking in the park at sunrise, oil painting style',400,50,7,'ddim'],
                    ['a cut teddy bear reading a book in the park, oil painting style, high quality',700,50,7,'ddim'],
                    ['an epic tornado attacking above a glowing city at night, the tornado is made of smoke, highly detailed',230,50,7,'ddim'],
                    ['a jar filled with fire, 4K video, 3D rendered, well-rendered',400,50,7,'ddim'],
                    ['a teddy bear walking in the park, oil painting style, high quality',400,50,7,'ddim'],
                    ['a teddy bear walking on the street, 2k, high quality',100,50,7,'ddim'],
                    ['a panda taking a selfie, 2k, high quality',400,50,7,'ddim'],
                    ['a polar bear playing drum kit in NYC Times Square, 4k, high resolution',400,50,7,'ddim'],
                    ['jungle river at sunset, ultra quality',400,50,7,'ddim'],
                    ['a shark swimming in clear Carribean ocean, 2k, high quality',400,50,7,'ddim'],
                    ['A steam train moving on a mountainside by Vincent van Gogh',230,50,7,'ddim'],
                    ['a confused grizzly bear in calculus class',1000,50,7,'ddim']],
            fn = infer,
            inputs=[prompt, seed_inp, ddim_steps,cfg,infer_type],
            outputs=[video_out],
            cache_examples=False,
        )
        ex.dataset.headers = [""]
         
    submit_btn.click(infer, inputs, outputs)

demo.queue(max_size=12).launch()


