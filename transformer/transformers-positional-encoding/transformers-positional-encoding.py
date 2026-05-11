import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length).reshape(seq_length, 1)
    div_term = np.array([10000**(-2*i/d_model) for i in range(d_model // 2)]).reshape(1, d_model // 2)
    theta = pos * div_term
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(theta)
    pe[:, 1::2] = np.cos(theta)
    return pe
    pass