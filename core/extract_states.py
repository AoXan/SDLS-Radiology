import torch
import pandas as pd
import argparse
import pathlib
import pickle
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel

def get_all_layer_states(model, tokenizer, processor, image_path, text, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except: return None

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            decoder_input_ids=inputs.input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # Locate last valid token
    last_token_idx = inputs.attention_mask.sum(dim=1) - 1
    
    # Extract states from ALL layers (Eq. 1)
    all_layers_mcv = []
    for layer_tensor in outputs.decoder_hidden_states:
        # layer_tensor: (batch, seq, dim)
        vec = layer_tensor[0, last_token_idx.item(), :].cpu().numpy()
        all_layers_mcv.append(vec)
    
    return all_layers_mcv

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Simplified loading for demo; use utils.model_loader in practice
    model = VisionEncoderDecoderModel.from_pretrained(args.model_ckpt).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    processor = AutoImageProcessor.from_pretrained(args.model_ckpt)

    df = pd.read_csv(args.input_csv)
    if args.sample_size: df = df.head(int(args.sample_size))
    
    results = []
    print("[Stage II] Extracting MCVs...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = pathlib.Path(args.image_root) / row['image_path']
        if not img_path.exists(): continue

        mcv_clean = get_all_layer_states(model, tokenizer, processor, img_path, row['impression_no_history'], device)
        mcv_hist = get_all_layer_states(model, tokenizer, processor, img_path, row['impression_with_history'], device)

        if mcv_clean is not None and mcv_hist is not None:
            results.append({
                'dicom_id': str(row['dicom_id']),
                'mcv_clean': mcv_clean,   # List[np.array]
                'mcv_history': mcv_hist   # List[np.array]
            })

    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--model_ckpt", default="IAMJB/chexpert-mimic-cxr-impression-baseline")
    parser.add_argument("--output_path", default="data/hidden_states.pkl")
    parser.add_argument("--sample_size", default=None)
    args = parser.parse_args()
    main(args)