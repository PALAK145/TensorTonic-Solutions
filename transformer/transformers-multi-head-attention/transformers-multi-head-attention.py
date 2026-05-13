import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # Step 1: Project — (batch, seq_len, d_model)
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Step 2: Split into heads — (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        # x: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_h = split_heads(Q_proj)  # (batch, num_heads, seq_len, d_k)
    K_h = split_heads(K_proj)
    V_h = split_heads(V_proj)

    # Step 3: Scaled dot-product attention per head
    # scores: (batch, num_heads, seq_len, seq_len)
    scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    attended = weights @ V_h  # (batch, num_heads, seq_len, d_k)

    # Step 4: Concatenate heads — (batch, seq_len, d_model)
    concat = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # Step 5: Output projection
    return concat @ W_o
    pass