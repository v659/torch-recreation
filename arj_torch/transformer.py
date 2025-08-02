from arj_torch.nn import *
import numpy as np
class SelfAttention:
    def __init__(self, embed_dim, num_heads=1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
        self.proj = Linear(embed_dim, embed_dim)  # projection after attention

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = len(x.data), len(x.data[0]), len(x.data[0][0])
        assert C == self.embed_dim, f"Expected input dim {self.embed_dim}, got {C}"

        # Step 1: Project input to q, k, v
        q = self.query(x)  # ArjTensor
        k = self.key(x)
        v = self.value(x)

        # Convert to NumPy
        q_np = np.array(q.data)  # [B, T, C]
        k_np = np.array(k.data)  # [B, T, C]
        v_np = np.array(v.data)  # [B, T, C]

        # Step 2: Compute attention scores via batch matmul
        scores = np.matmul(q_np, np.transpose(k_np, (0, 2, 1))) / math.sqrt(C)
        mask = np.tril(np.ones((T, T), dtype=bool))  # Lower triangle
        scores = np.where(mask[None, :, :], scores, -np.inf)

        # Step 3: Apply softmax (numerical stability)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores)
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)  # [B, T, T]

        # Step 4: Apply attention weights to v
        out_np = np.matmul(attn_weights, v_np)  # [B, T, C]
        # Step 5: Convert back to ArjTensor
        out_tensor = ArjTensor(out_np.tolist(), requires_grad=True, _children=(x,), _op="attn")

        # Step 6: Final linear projection
        proj_out = self.proj(out_tensor)

        # Final shape validation
        for row in proj_out.data:
            for token in row:
                assert len(
                    token) == self.embed_dim, f"Projected output dim mismatch: expected {self.embed_dim}, got {len(token)}"


        return proj_out

    def parameters(self):
        return (
            self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
            + self.proj.parameters()
        )


# === Transformer Block ===
class TransformerBlock:
    def __init__(self, embed_dim, num_heads=1):
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, embed_dim * 4)

        self.norm2 = LayerNorm(embed_dim)  # Must match final FFN output dim

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)

        for row in x.data:
            for token in row:
                assert len(token) == self.norm2.dim, f"LayerNorm expected dim={self.norm2.dim}, got {len(token)}"

        x = self.norm2(x + ff_out)
        return x

    def parameters(self):
        return self.attn.parameters() + self.norm1.parameters() + self.ff.parameters() + self.norm2.parameters()


# === Full Model ===
class PatternTransformer:
    def __init__(self, seq_len=4, embed_dim=16, heads=1, layers=1):
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.layers = layers

        self.input_proj = Linear(in_features=embed_dim, out_features=embed_dim)

        # Create multiple Transformer blocks
        self.blocks = [TransformerBlock(embed_dim, heads) for _ in range(layers)]

        self.output_proj = Linear(embed_dim, 1)  # Optional: used in some tasks

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        x_np = np.array(x.data)
        if x_np.ndim == 2:
            x_np = np.expand_dims(x_np, axis=-1)  # (B, T) → (B, T, 1)
        x = ArjTensor(x_np, requires_grad=True)


        x = self.input_proj(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        return x

    def parameters(self):
        params = self.input_proj.parameters() + self.output_proj.parameters()
        for block in self.blocks:
            params += block.parameters()
        return params

