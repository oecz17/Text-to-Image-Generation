import torch
import torchvision
import numpy as np
import gradio as gr
from parti_pytorch import Parti, VitVQGanVAE
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

title = "Text-to-Image Generation"
description = "Generate images with Parti or Stable Diffusion model"

model_id = "OFA-Sys/small-stable-diffusion-v0"
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
	model_id, 
	torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
	scheduler=scheduler)

vit_vae = VitVQGanVAE(
	dim=128,
	image_size=128,
	patch_size=12,
	num_layers=4,
	)

vit_vae.load_state_dict(torch.load('models/vae.1250.pt', map_location=torch.device('cpu'))) 

parti = Parti(
    vae = vit_vae, 
    dim = 128,             
    depth = 12,                
    dim_head = 64,      
    heads = 8,             
    dropout = 0.,           
    cond_drop_prob = 0.25,  
    ff_mult = 5,              
    t5_name = 't5-large',    
)

# I had to add this because from library parti_pytorch, torch.save() function
# was overriden to avoid saving vgg weights for some reason
vgg_st_dict = torchvision.models.vgg16(pretrained = True).state_dict()
updated_st_dict = {}
for key,value in vgg_st_dict.items():
	if key != 'classifier.6.weight' and key!='classifier.6.bias':
		updated_key = 'vae.vgg.'+key
		updated_st_dict[updated_key] = value

parti_st_dict = torch.load('models/parti.pt', map_location=torch.device('cpu'))
combined_state_dict = parti_st_dict.copy()
combined_state_dict.update(updated_st_dict)

parti.load_state_dict(combined_state_dict) 

def text_to_image(prompt, model_type):
	if model_type == 'Diffusers':
		img = pipe(prompt,width=200, height=200).images[0]
		img = np.array(img)
		return img
	else:
		img = parti.generate(texts = [prompt], cond_scale=3., return_pil_images=True)[0].resize((200,200))
		return np.array(img)

demo = gr.Interface(
	fn=text_to_image, 
	inputs=['text', gr.Radio(["Parti", "Diffusers"])], 
	outputs='image',
	title=title,
	description=description)

demo.launch()