from arj_torch.nn import *


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

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention score computation
        scores = []
        for b in range(B):
            score = []
            for i in range(T):
                row = []
                for j in range(T):
                    dot = sum(q.data[b][i][d] * k.data[b][j][d] for d in range(C))
                    row.append(dot / math.sqrt(C))
                score.append(row)
            scores.append(score)

        # Softmax attention weights
        attn_weights = []
        for score in scores:
            soft = []
            for row in score:
                exps = [math.exp(s) for s in row]
                s = sum(exps)
                soft.append([e / s for e in exps])
            attn_weights.append(soft)

        # Apply attention weights to value vectors
        out = []
        for b in range(B):
            batch_out = []
            for i in range(T):
                out_vec = [0.0] * C
                for j in range(T):
                    for d in range(C):
                        out_vec[d] += attn_weights[b][i][j] * v.data[b][j][d]
                batch_out.append(out_vec)
            out.append(batch_out)

        out_tensor = ArjTensor(out, requires_grad=True, _children=(x,), _op="attn")

        # Apply final projection
        proj_out = self.proj(out_tensor)

        # ✅ Final shape check (important!)
        for row in proj_out.data:
            for token in row:
                assert len(token) == self.embed_dim, f"Projected output dim mismatch: expected {self.embed_dim}," \
                                                     f" got {len(token)}"

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
        self.ff = Sequential(
            Linear(embed_dim, embed_dim * 2),  # 16 → 32
            ReLU(),
            Linear(embed_dim * 2, embed_dim)  # 32 → 16 ✅
        )
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
    def __init__(self, seq_len=4, embed_dim=16, heads=1):
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Project scalar inputs (1D per token) to embedding dim
        self.input_proj = Linear(in_features=1, out_features=embed_dim)
        self.transformer = TransformerBlock(embed_dim, heads)
        self.output_proj = Linear(embed_dim, 1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # x: (B, T)
        # Convert (B, T) → (B, T, 1)
        x = ArjTensor([[[v] for v in row] for row in x.data], requires_grad=True)

        # Apply input_proj to each (B, T, 1) → (B, T, embed_dim)
        projected = []

        for row in x.data:
            proj_row = []
            for v in row:
                proj_v = self.input_proj(ArjTensor([[v[0]]], requires_grad=True))
                assert len(proj_v.data[
                               0]) == self.embed_dim, f"Expected embedding dim={self.embed_dim}," \
                                                      f" got {len(proj_v.data[0])}"

                proj_row.append(proj_v.data[0])
            projected.append(proj_row)
        x = ArjTensor(projected, requires_grad=True)

        # Transformer + output
        x = self.transformer(x)                   # (B, T, embed_dim)
        x = [row[-1] for row in x.data]           # use last token → (B, embed_dim)
        x = ArjTensor(x, requires_grad=True)
        return self.output_proj(x)                # (B, 1)

    def parameters(self):
        return self.input_proj.parameters() + self.transformer.parameters() + self.output_proj.parameters()
