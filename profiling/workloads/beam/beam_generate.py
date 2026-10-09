#!/usr/bin/env python3
"""Stand-in for LLM tree search: Hugging Face generate() with beam search on a small causal LM.  The batch x
num_beams candidates (Q) share the weights and their KV-cache prefixes; beam selection (log-softmax, top-k over
batch x beams x vocab, cache reorder) is the symbolic part, the transformer forward the neural part.  Untested."""
import argparse, time, torch  # noqa: E401
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--num-beams", type=int, default=4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--prompt-len", type=int, default=32)
ap.add_argument("--new-tokens", type=int, default=64)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(0); dt_ = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt_).to(dev).eval()
ids = torch.randint(0, tok.vocab_size, (a.batch, a.prompt_len), device=dev)
kw = dict(attention_mask=torch.ones_like(ids), num_beams=a.num_beams, num_return_sequences=1, do_sample=False,
          max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, early_stopping=False, pad_token_id=tok.eos_token_id)
with torch.no_grad():
    model.generate(ids, **kw); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(a.iters):
        torch.cuda.nvtx.range_push("generate_beam"); out = model.generate(ids, **kw); torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
dt = time.perf_counter() - t0
cfg = model.config
params = sum(p.numel() * p.element_size() for p in model.parameters())
kv = 2 * cfg.n_layer * cfg.n_embd * (a.prompt_len + a.new_tokens) * a.batch * a.num_beams * torch.finfo(dt_).bits // 8
print(f"SYMFABRIC_METRIC W_bytes {params + kv}")    # weights + KV cache at the final length (GPT-2 config fields)
print(f"SYMFABRIC_METRIC Q {a.batch * a.num_beams}\nSYMFABRIC_METRIC hot_time_s {dt:.4f}")
print(f"SYMFABRIC_METRIC tokens_per_s {a.iters * a.batch * a.new_tokens / dt:.1f}")
print(f"SYMFABRIC_METRIC output_shape {'x'.join(map(str, out.shape))}")
