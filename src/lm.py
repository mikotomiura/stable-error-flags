"""Wrapper for the instruction-tuned LLMs (Qwen, OLMo). Preregistration v4, Sec. 3: bf16, greedy decoding,
and probabilities obtained from the log-likelihoods of candidate strings (no sampling).

- choose(): sums the log-likelihood of each candidate string followed by the end-of-turn token
  (tokenizer.eos_token) and normalizes over the candidates. It also decodes greedily (up to max_new_tokens);
  an output is "parsed" if it matches exactly one candidate after normalize() (strip whitespace, quotes and
  trailing punctuation; lowercase).
- generate(): greedy decoding.
Every call logs the number of input and output tokens and the wall-clock time (reported per Sec. 5).
"""
from __future__ import annotations

import math
import re
import time

KV_BUDGET = 3 * 1024 ** 3  # bytes of copied prefix cache per candidate batch


def normalize(s: str) -> str:
    s = s.strip().strip("\"'`*").strip()
    s = re.sub(r"[.\u3002!?,;:]+$", "", s).strip().strip("\"'`*").strip()
    return s.lower()


class ChatLM:
    def __init__(self, path, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.eos_id = self.tok.convert_tokens_to_ids(self.tok.eos_token)
        self.log = []  # one record per call

    def prompt_ids(self, user_text: str):
        ids = self.tok.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True,
                                           tokenize=True, return_dict=False)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)

    def _forward_prefix(self, ids):
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=x, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def choose(self, user_text: str, candidates: list[str], max_new_tokens=8, tag=""):
        """Return per-candidate log-likelihoods, normalized probabilities, the greedy output, and the parse flag."""
        import copy
        torch = self.torch
        t0 = time.monotonic()
        ids = self.prompt_ids(user_text)
        last_logits, past = self._forward_prefix(ids)
        cand_ids = [self.tok.encode(c, add_special_tokens=False) + [self.eos_id] for c in candidates]
        # Score the candidates in batches on top of the shared prefix cache. Right padding does not affect real
        # tokens under causal attention. The prefix cache is copied once per candidate, so a batch is limited to
        # KV_BUDGET bytes of copied cache; models with few key-value heads (Qwen) always fit in one batch.
        L = max(len(c) for c in cand_ids)
        pad = self.eos_id
        batch = torch.tensor([c + [pad] * (L - len(c)) for c in cand_ids], device=self.device)
        logits = None
        if L > 1:
            per = max(1, int(KV_BUDGET // max(1, self.kv_bytes_per_token() * len(ids))))
            parts = []
            for s0 in range(0, len(cand_ids), per):
                sub = batch[s0: s0 + per]
                pkv = copy.deepcopy(past)
                pkv.batch_repeat_interleave(len(sub))
                with torch.inference_mode():
                    parts.append(self.model(input_ids=sub[:, :-1], past_key_values=pkv, use_cache=True).logits.float())
                del pkv
            logits = torch.cat(parts, 0)
        first_lp = torch.log_softmax(last_logits, -1)
        lls = []
        for j, c in enumerate(cand_ids):
            ll = float(first_lp[c[0]])
            if len(c) > 1:
                lp = torch.log_softmax(logits[j, : len(c) - 1], -1)
                ll += float(sum(lp[k, c[k + 1]] for k in range(len(c) - 1)))
            lls.append(ll)
        m = max(lls)
        z = sum(math.exp(v - m) for v in lls)
        probs = [math.exp(v - m) / z for v in lls]
        # Greedy decoding from the same prefix cache.
        gen, n_gen = self._greedy_from(ids, last_logits, past, max_new_tokens)
        norm = normalize(gen)
        hits = [i for i, c in enumerate(candidates) if normalize(c) == norm]
        rec = {"tag": tag, "kind": "choose", "in_tokens": len(ids), "out_tokens": n_gen,
               "seconds": time.monotonic() - t0}
        self.log.append(rec)
        return {"logliks": lls, "probs": probs, "greedy": gen, "parsed": len(hits) == 1,
                "greedy_choice": hits[0] if len(hits) == 1 else None}

    def kv_bytes_per_token(self):
        c = self.model.config
        heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return 2 * c.num_hidden_layers * heads * head_dim * self.model.dtype.itemsize

    def _greedy_from(self, ids, last_logits, past, max_new_tokens):
        torch = self.torch
        out_ids = []
        nxt = int(torch.argmax(last_logits))
        pkv = past
        for _ in range(max_new_tokens):
            if nxt == self.eos_id or nxt in self._stop_ids():
                break
            out_ids.append(nxt)
            with torch.inference_mode():
                o = self.model(input_ids=torch.tensor([[nxt]], device=self.device), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            nxt = int(torch.argmax(o.logits[0, -1]))
        # The count of generated tokens is returned with the text: decoding and re-encoding can change the
        # count, and the token limit is judged on the tokens actually generated.
        return self.tok.decode(out_ids, skip_special_tokens=True), len(out_ids)

    def _stop_ids(self):
        if not hasattr(self, "_stops"):
            gc = getattr(self.model, "generation_config", None)
            e = getattr(gc, "eos_token_id", None) if gc else None
            self._stops = set(e if isinstance(e, list) else ([e] if e is not None else [])) | {self.eos_id}
        return self._stops

    def generate(self, user_text: str, max_new_tokens: int, tag=""):
        t0 = time.monotonic()
        ids = self.prompt_ids(user_text)
        last_logits, past = self._forward_prefix(ids)
        gen, n_out = self._greedy_from(ids, last_logits, past, max_new_tokens)
        self.log.append({"tag": tag, "kind": "generate", "in_tokens": len(ids), "out_tokens": n_out, "seconds": time.monotonic() - t0})
        return {"text": gen, "hit_limit": n_out >= max_new_tokens}
