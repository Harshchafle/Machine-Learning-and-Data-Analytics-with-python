
from google.generativeai import genai
from google.genai import types

client = genai.Client()

op = client.models.generate_video(
    model="veo-3.1-generate-preview",
    prompt = prompt,
    config = types.GenerateVideoConfig(
        reference_images=[img1, img2]
    )
)