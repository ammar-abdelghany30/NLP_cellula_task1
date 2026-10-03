import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize model and processor (CPU compatible)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(_device)

def generate_caption(image_input) -> str:
    """
    Generates a descriptive text caption from an input image.
    Accepts a PIL Image or file path.
    """
    if isinstance(image_input, str):
        raw_image = Image.open(image_input).convert('RGB')
    else:
        raw_image = Image.open(image_input).convert('RGB')
        
    inputs = _processor(raw_image, return_tensors="pt").to(_device)
    out = _model.generate(**inputs, max_new_tokens=50)
    caption = _processor.decode(out[0], skip_special_tokens=True)
    return caption