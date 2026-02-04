import numpy as np
import pickle
import argparse
import pathlib
from sklearn.decomposition import PCA

def compute_vector(vectors: np.ndarray, method: str = 'qr') -> np.ndarray:
    """
    Computes SDIV (QR) or Global ICV (PCA).
    """
    if method == 'qr':
        # Eq. 5: Mean of Orthogonal Basis
        print("[Math] Method: QR (SDIV)")
        Q, _ = np.linalg.qr(vectors.T)
        vec = Q.mean(axis=1)
    elif method == 'pca':
        # Eq. 3: Top-1 Principal Component
        print("[Math] Method: PCA (Global ICV)")
        pca = PCA(n_components=1)
        pca.fit(vectors)
        vec = pca.components_[0]
    else:
        raise ValueError(f"Unknown method: {method}")
        
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec

def main(args):
    with open(args.input_pkl, 'rb') as f:
        data = pickle.load(f)

    deltas = []
    print(f"[Stage II] Computing deltas for {len(data)} pairs...")
    
    for item in data:
        z_hist = item['mcv_history']
        z_clean = item['mcv_clean']
        
        # Flatten multi-layer lists (Eq. 1 Compliance)
        if isinstance(z_hist, list): z_hist = np.concatenate(z_hist)
        if isinstance(z_clean, list): z_clean = np.concatenate(z_clean)
        
        deltas.append(z_hist - z_clean)

    V = np.vstack(deltas)
    vector = compute_vector(V, method=args.method)
    
    np.save(args.output_path, vector)
    print(f"[Success] Vector saved to {args.output_path} (Method: {args.method})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True)
    parser.add_argument("--output_path", default="core/sdiv_vector.npy")
    parser.add_argument("--method", choices=['qr', 'pca'], default='qr', help="qr for SDIV, pca for Global ICV")
    args = parser.parse_args()
    main(args)