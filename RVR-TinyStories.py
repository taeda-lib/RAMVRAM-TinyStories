import torch
import torch.nn as nn
import taeda

torch.manual_seed(0)

# ------------------------------------------------------------
# 1. Data: a tiny corpus, tokenized per character
# ------------------------------------------------------------
corpus = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the sun is hot and the sky is blue. "
    "a small robot likes to read books. "
    "the robot reads about cats and dogs. "
    "the cat and the dog are good friends. "
    "at night the moon is bright and the stars are far. "
    "the robot dreams of the sun and the sky. "
)

chars = sorted(set(corpus))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)

CTX = 16      # context window (characters)
EMB = 32      # embedding size -> CTX * EMB = 512 inputs to first Linear

# All (context -> next char) training pairs
starts = torch.arange(0, len(data) - CTX)
X_all = torch.stack([data[s:s + CTX] for s in starts])
Y_all = torch.stack([data[s + CTX] for s in starts])

# ------------------------------------------------------------
# 2. Model: nn.Sequential (required by RAMVRAMModel)
# ------------------------------------------------------------
model = nn.Sequential(
    nn.Embedding(vocab, EMB),
    nn.Flatten(),                 # (B, CTX, EMB) -> (B, CTX*EMB = 512)
    nn.Linear(CTX * EMB, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, vocab),
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = taeda.RAMVRAMModel(
    model,
    device=device,
    max_vram_percent=80
)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
STEPS = 1500
BATCH = 64

model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(X_all), (BATCH,))
    x, y = X_all[idx], Y_all[idx]

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)
    loss = criterion(logits, y.to(logits.device))

    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == 1:
        print(f"Step {step:4d}: loss {loss.item():.4f}")

# ------------------------------------------------------------
# 4. Generate
# ------------------------------------------------------------
@torch.no_grad()
def generate(prompt, n_chars=200, temperature=0.7):
    model.eval()

    ids = [stoi[c] for c in prompt if c in stoi]
    ids = ([stoi[" "]] * CTX + ids)[-CTX:]   # left-pad / trim to CTX

    out = list(prompt)

    for _ in range(n_chars):
        x = torch.tensor([ids[-CTX:]], dtype=torch.long)
        logits = model(x)[0] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()

        ids.append(nxt)
        out.append(itos[nxt])

    return "".join(out)


print("\n=== Generated text ===")
print(generate("the robot ", n_chars=200))
print()
print(generate("the cat ", n_chars=120))

print()
model.report()
