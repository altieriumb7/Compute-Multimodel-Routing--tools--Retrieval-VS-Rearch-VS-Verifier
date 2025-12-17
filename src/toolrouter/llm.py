from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class LLM:
    model_name_or_path: str
    max_new_tokens: int = 128
    device: str | None = None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def answer(self, question: str, docs: List[dict], system: Optional[str] = None) -> str:
        ctx = "\n\n".join([f"[{i}] {d.get('title','')}\n{d.get('text','')}" for i, d in enumerate(docs, 1)])
        prompt = ""
        if system:
            prompt += f"{system.strip()}\n\n"
        prompt += (
            "Answer the question using ONLY the provided context. "
            "If the answer isn't in the context, say 'I don't know'.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return suffix after "Answer:" if possible
        if "Answer:" in text:
            return text.split("Answer:", 1)[1].strip()
        return text.strip()
