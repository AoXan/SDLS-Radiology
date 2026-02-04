import argparse
import torch
import pandas as pd
import pathlib
import sys
from tqdm.auto import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.model_loader import load_model, prepare_inputs
from core.steering import SDLSEngine

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model & Data
    model, tokenizer, processor = load_model(args.backend, args.model_path, device)
    df = pd.read_csv(args.input_csv)
    if args.sample_size: df = df.head(int(args.sample_size))

    # Setup Steering
    engine = SDLSEngine(model, device)
    if args.use_sdiv:
        engine.load_vector(args.icv_path, args.strength)
        layers = list(range(args.layer_start, args.layer_end))
        # Support submodule selection
        submodule = 'attention' if 'Attention' in args.strategy else 'layer'
        strat_name = 'SteerFair' if 'SteerFair' in args.strategy else args.strategy
        engine.activate(layers, strat_name, submodule)

    results = []
    print(f"[Run] Generating ({args.backend})...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = pathlib.Path(args.image_root) / row['image_path']
        if not img_path.exists(): continue
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            
            # LLaVA Prompt Fix
            prompt = "<image>\nDescribe the findings." if args.backend == 'llava_med' else "Impression:"
            
            inputs = prepare_inputs(args.backend, processor, tokenizer, image, prompt, device)
            
            with torch.no_grad():
                if args.backend == 'llava_med':
                    out = model.generate(**inputs, max_new_tokens=100)
                    text = processor.decode(out[0], skip_special_tokens=True)
                else:
                    # BiomedGPT parameter fix
                    img_arg = {'patch_images': inputs['pixel_values']} if args.backend == 'biomedgpt' else {'pixel_values': inputs['pixel_values']}
                    out = model.generate(**img_arg, max_length=120, num_beams=4, early_stopping=True)
                    text = tokenizer.decode(out[0], skip_special_tokens=True)

            results.append({
                'dicom_id': row['dicom_id'],
                'generated_report': text,
                'impression_with_history': row.get('impression_with_history', '')
            })
        except Exception as e:
            print(f"Error: {e}")

    engine.reset()
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--use_sdiv", action="store_true")
    parser.add_argument("--icv_path")
    parser.add_argument("--strength", type=float, default=-2.0)
    parser.add_argument("--strategy", default="SteerFair_Attention") # e.g. SteerFair_Attention
    parser.add_argument("--layer_start", type=int, default=2)
    parser.add_argument("--layer_end", type=int, default=10)
    parser.add_argument("--sample_size", type=int)
    args = parser.parse_args()
    main(args)