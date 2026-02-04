import torch
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel, LlavaForConditionalGeneration, AutoProcessor

def load_model(backend, model_path, device):
    print(f"[Loader] Loading {backend} from {model_path}...")
    if backend == 'iamjb':
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
    elif backend == 'biomedgpt':
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
    elif backend == 'llava_med':
        model = LlavaForConditionalGeneration.from_pretrained(model_path).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    return model, tokenizer, processor

def prepare_inputs(backend, processor, tokenizer, image, text, device):
    if backend == 'llava_med':
        return processor(text=text, images=image, return_tensors="pt").to(device)
    else:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        return {'pixel_values': pixel_values}