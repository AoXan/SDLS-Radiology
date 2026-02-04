import sys
import torch
from types import SimpleNamespace
from importlib.util import spec_from_loader
from transformers import (
    VisionEncoderDecoderModel,
    BertTokenizer,
    ViTImageProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    OFAForConditionalGeneration,
    OFATokenizer
)

# Mock bitsandbytes to prevent import errors in certain environments
try:
    import bitsandbytes
except ImportError:
    class MockBitsAndBytes:
        __spec__ = spec_from_loader("bitsandbytes", loader=None)
    sys.modules["bitsandbytes"] = MockBitsAndBytes()

# Register safe globals for PyTorch 2.6+ serialization compatibility
try:
    torch.serialization.add_safe_globals([SimpleNamespace, set])
except (AttributeError, ImportError):
    pass

def load_model(backend: str, model_path: str, device: str):
    print(f"[Model Loader] Loading backend: {backend} from {model_path}...")
    
    if backend == 'iamjb':
        tokenizer = BertTokenizer.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device).eval()
        
    elif backend == 'biomedgpt':
        try:
            tokenizer = OFATokenizer.from_pretrained(model_path)
            model = OFAForConditionalGeneration.from_pretrained(model_path).to(device).eval()
            # OFA uses ViT-style preprocessing for 224x224 input
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        except Exception as e:
            raise RuntimeError(f"Failed to load BiomedGPT (OFA). Error: {e}")

    elif backend == 'llava_med':
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        ).to(device).eval()
        
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    return model, tokenizer, processor

def prepare_inputs(backend: str, processor, tokenizer, image, text, device):
    if backend == 'iamjb':
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=128
        ).to(device)
        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': inputs.input_ids
        }

    elif backend == 'biomedgpt':
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(device)
        return {
            'pixel_values': pixel_values,
            'input_ids': inputs.input_ids
        }

    elif backend == 'llava_med':
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        return inputs

    else:
        raise ValueError(f"Unknown backend: {backend}")