"""Single-pass greedy generation with confidence and hidden-state-shift capture."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationTrace:
    answer: str
    token_ids: list[int]
    token_pieces: list[str]
    confidences: list[float]
    shifts: list[float | None]
    elapsed_seconds: float


def choose_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


class WhiteBoxGenerator:
    """Minimal Hugging Face causal-LM wrapper used by the experiment."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str = "main",
        device: str = "auto",
        layer: int = -1,
        prompt_style: str = "instruct",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = choose_device(torch, device)
        self.layer = layer
        self.prompt_style = prompt_style
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        self.eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        self.eos_ids.discard(None)

    def _prompt_ids(self, question: str):
        if self.prompt_style == "base":
            prompt = (
                f"Question: {question}\n"
                "Answer with one short factual sentence and no explanation:"
            )
            return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if self.prompt_style != "instruct":
            raise ValueError(f"unsupported prompt style: {self.prompt_style}")
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer factual questions directly and state your best answer even if "
                    "uncertain. Follow the requested sentence frame exactly and add no facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question}\nReplace only the bracketed text in this frame: "
                    "'The requested answer is [short answer], stated as my best factual "
                    "response.' Add no explanation or supporting detail."
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def generate(self, question: str, *, max_new_tokens: int) -> GenerationTrace:
        """Greedily generate once and record the state used to choose each token.

        At step ``t``, confidence and hidden state are both taken from the model output
        that predicts token ``t``. The first generated step has no preceding state, so
        its shift is ``None``. KV caching keeps this a single autoregressive pass.
        """
        torch = self.torch
        prompt_ids = self._prompt_ids(question)
        token_ids: list[int] = []
        confidences: list[float] = []
        shifts: list[float | None] = []
        previous_hidden = None

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=prompt_ids,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            current_hidden = outputs.hidden_states[self.layer][:, -1, :]

            for _ in range(max_new_tokens):
                next_token = torch.argmax(logits, dim=-1)
                next_token_id = int(next_token.item())
                if next_token_id in self.eos_ids:
                    break

                log_probability = torch.log_softmax(logits.float(), dim=-1)[
                    0, next_token_id
                ]
                confidence = float(torch.exp(log_probability).item())
                if previous_hidden is None:
                    shift = None
                else:
                    numerator = torch.linalg.vector_norm(
                        current_hidden.float() - previous_hidden.float()
                    )
                    denominator = torch.linalg.vector_norm(previous_hidden.float()) + 1e-8
                    shift = float((numerator / denominator).item())

                token_ids.append(next_token_id)
                confidences.append(confidence)
                shifts.append(shift)
                previous_hidden = current_hidden.detach()

                outputs = self.model(
                    input_ids=next_token.reshape(1, 1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                current_hidden = outputs.hidden_states[self.layer][:, -1, :]

        elapsed = time.perf_counter() - start
        return GenerationTrace(
            answer=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            token_ids=token_ids,
            token_pieces=[
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids
            ],
            confidences=confidences,
            shifts=shifts,
            elapsed_seconds=elapsed,
        )
