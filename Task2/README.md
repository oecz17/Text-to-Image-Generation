# Installation

```bash
pip install gradio
pip install diffusers
pip install accelerate
pip install parti-pytorch
pip install transformers
pip install sentencepiece
```

## Usage

Just run:
```bash
gradio app.py
```
In the app I added two models:
- Parti (trained on task 1)
- Distilled Stable Diffusion model

It takes a while the first time the app is run because it will download the model's weights.
Once the model is selected and the prompt is added, you just click on send and the app will show the prediction.