import torch
import torch.nn as nn
import numpy as np
from typing import List

class SDLSEngine:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.hooks = []
        self.sdiv = None
        self.strength = 0.0

    def load_vector(self, vector_path: str, strength: float):
        vec = np.load(vector_path)
        if vec.ndim > 1: vec = vec.flatten() # Handle potential shape issues
        self.sdiv = torch.tensor(vec, dtype=torch.float32, device=self.device)
        self.strength = strength

    def _get_hook_fn(self, strategy: str):
        def hook(module, input, output):
            # Output is typically (Batch, Seq, Dim) or tuple
            h = output[0] if isinstance(output, tuple) else output
            
            # Eq. 6: Norm-Preserving Addition
            norm = torch.norm(h, p=2, dim=-1, keepdim=True)
            h_normalized = h / (norm + 1e-12)
            
            if strategy == 'GentleInject':
                # Apply only to first token (CLS/BOS)
                # Reshape sdiv to broadcast: (1, 1, Dim)
                delta = (self.strength * self.sdiv).view(1, 1, -1)
                h[:, 0, :] = h_normalized[:, 0, :] + delta
            else:
                # Apply to all tokens (SteerFair/Global)
                delta = (self.strength * self.sdiv).view(1, 1, -1)
                h = h_normalized + delta
                
            # Restore norm
            h_new = h * norm
            
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook

    def activate(self, layers: List[int], strategy: str = 'SteerFair', submodule: str = 'layer'):
        self.reset()
        
        # 1. ICV-Token Strategy (Input Embedding Injection)
        if strategy == 'ICV-Token':
            embed_module = self.model.get_input_embeddings()
            def token_hook(module, input, output):
                # Add to first token
                vec = self.sdiv.to(output.device).to(output.dtype)
                output[:, 0, :] += (self.strength * vec)
                return output
            self.hooks.append(embed_module.register_forward_hook(token_hook))
            return

        # 2. Hidden State Injection
        # Auto-detect layers
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            decoder_layers = self.model.decoder.layers # BiomedGPT
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            decoder_layers = self.model.model.layers # LLaMA
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            decoder_layers = self.model.bert.encoder.layer # IAMJB
        else:
            raise ValueError("Unknown model architecture.")

        print(f"[Steering] Active: {strategy} on {submodule} of layers {layers}")

        for idx in layers:
            block = decoder_layers[idx]
            target = block
            
            if submodule == 'attention':
                # Heuristic to find self-attention module
                if hasattr(block, 'self_attn'): target = block.self_attn
                elif hasattr(block, 'attention'): target = block.attention
                elif hasattr(block, 'self_attention'): target = block.self_attention
            
            self.hooks.append(target.register_forward_hook(self._get_hook_fn(strategy)))

    def reset(self):
        for h in self.hooks: h.remove()
        self.hooks = []