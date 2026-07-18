"""Autoregressive generation with token-distribution and hidden-state signals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationTrace:
    answer: str
    token_ids: list[int]
    token_pieces: list[str]
    confidences: list[float]
    shifts: list[float | None]
    token_margins: list[float]
    token_entropies: list[float]
    hidden_cosine_distances: list[float | None]
    hidden_norms: list[float]
    mean_hidden: list[float]
    elapsed_seconds: float


@dataclass(frozen=True)
class BinaryJudgmentTrace:
    yes_probability: float
    no_probability: float
    normalized_no_probability: float
    yes_token_id: int
    no_token_id: int
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
            prompt = f"Question: {question}\nAnswer:"
            return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if self.prompt_style != "instruct":
            raise ValueError(f"unsupported prompt style: {self.prompt_style}")
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer factual questions directly. If uncertain, give your best answer. "
                    "Keep the response brief."
                ),
            },
            {
                "role": "user",
                "content": f"{question}\nGive only the short answer, with no explanation.",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def _grounded_prompt_ids(self, knowledge: str, question: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied knowledge. Do not add unsupported "
                    "facts. Give only the direct answer."
                ),
            },
            {
                "role": "user",
                "content": f"Knowledge:\n{knowledge}\n\nQuestion:\n{question}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def _judge_prompt_ids(self, instruction: str):
        if self.prompt_style == "base":
            prompt = f"{instruction}\nReply only Yes or No.\nVerdict:"
            return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        messages = [
            {
                "role": "system",
                "content": (
                    "Judge the proposed factual answer. Follow the requested binary "
                    "response exactly and add no explanation."
                ),
            },
            {"role": "user", "content": f"{instruction}\nReply only Yes or No."},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def _binary_token_ids(self) -> tuple[int, int]:
        yes_text, no_text = (" Yes", " No") if self.prompt_style == "base" else ("Yes", "No")
        yes_ids = self.tokenizer.encode(yes_text, add_special_tokens=False)
        no_ids = self.tokenizer.encode(no_text, add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError("binary judge requires single-token Yes and No encodings")
        return int(yes_ids[0]), int(no_ids[0])

    def _score_binary_judgment(self, instruction: str) -> BinaryJudgmentTrace:
        torch = self.torch
        input_ids = self._judge_prompt_ids(instruction)
        yes_token_id, no_token_id = self._binary_token_ids()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            probabilities = torch.softmax(outputs.logits[0, -1, :].float(), dim=-1)
            yes_probability = float(probabilities[yes_token_id].item())
            no_probability = float(probabilities[no_token_id].item())
        denominator = yes_probability + no_probability
        normalized_no = no_probability / denominator if denominator else 0.5
        return BinaryJudgmentTrace(
            yes_probability=yes_probability,
            no_probability=no_probability,
            normalized_no_probability=normalized_no,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            elapsed_seconds=time.perf_counter() - started,
        )

    def score_correctness(self, question: str, answer: str) -> BinaryJudgmentTrace:
        """Return P(No | fixed correctness prompt), normalized over Yes and No."""
        return self._score_binary_judgment(
            f"Question: {question}\nProposed answer: {answer}\n"
            "Is the proposed answer factually correct?"
        )

    def score_equivalence(
        self, question: str, answer_a: str, answer_b: str
    ) -> BinaryJudgmentTrace:
        """Return P(No | fixed same-answer prompt), normalized over Yes and No."""
        return self._score_binary_judgment(
            f"Question: {question}\nAnswer A: {answer_a}\nAnswer B: {answer_b}\n"
            "Do Answer A and Answer B give the same factual answer?"
        )

    def generate(
        self,
        question: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> GenerationTrace:
        """Generate once and record the state used to choose each token.

        At step ``t``, confidence and hidden state are both taken from the model output
        that predicts token ``t``. The first generated step has no preceding state, so
        its shift is ``None``. KV caching keeps this a single autoregressive pass.
        """
        torch = self.torch
        prompt_ids = self._prompt_ids(question)
        token_ids: list[int] = []
        confidences: list[float] = []
        shifts: list[float | None] = []
        token_margins: list[float] = []
        token_entropies: list[float] = []
        hidden_cosine_distances: list[float | None] = []
        hidden_norms: list[float] = []
        previous_hidden = None
        hidden_sum = None
        cpu_generator = None
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            if top_k < 2:
                raise ValueError("top_k must be at least 2")
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(0 if seed is None else seed)

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
                if do_sample:
                    sample_count = min(top_k, int(logits.shape[-1]))
                    top_values, top_indices = torch.topk(
                        logits.float() / temperature, k=sample_count, dim=-1
                    )
                    sample_probabilities = torch.softmax(top_values[0].cpu(), dim=-1)
                    cumulative = torch.cumsum(sample_probabilities, dim=-1)
                    keep = cumulative - sample_probabilities < top_p
                    filtered = sample_probabilities * keep
                    filtered = filtered / filtered.sum()
                    sampled_position = int(
                        torch.multinomial(
                            filtered, 1, generator=cpu_generator
                        ).item()
                    )
                    next_token_id = int(top_indices[0, sampled_position].item())
                    next_token = torch.tensor([next_token_id], device=self.device)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                    next_token_id = int(next_token.item())
                if next_token_id in self.eos_ids:
                    break

                log_probabilities = torch.log_softmax(logits.float(), dim=-1)
                top_probabilities = torch.exp(torch.topk(log_probabilities, k=2, dim=-1).values)
                confidence = float(torch.exp(log_probabilities[0, next_token_id]).item())
                token_margin = float((top_probabilities[0, 0] - top_probabilities[0, 1]).item())
                probabilities = torch.exp(log_probabilities)
                token_entropy = float(
                    (-(probabilities * log_probabilities).sum() / math.log(logits.shape[-1])).item()
                )
                current_float = current_hidden.float()
                current_vector = current_float.reshape(-1)
                hidden_sum = (
                    current_vector.detach().clone()
                    if hidden_sum is None
                    else hidden_sum + current_vector.detach()
                )
                hidden_norm = float(
                    (torch.linalg.vector_norm(current_float) / math.sqrt(current_float.shape[-1])).item()
                )
                if previous_hidden is None:
                    shift = None
                    hidden_cosine_distance = None
                else:
                    numerator = torch.linalg.vector_norm(
                        current_float - previous_hidden.float()
                    )
                    denominator = torch.linalg.vector_norm(previous_hidden.float()) + 1e-8
                    shift = float((numerator / denominator).item())
                    cosine = torch.nn.functional.cosine_similarity(
                        current_float, previous_hidden.float(), dim=-1, eps=1e-8
                    )
                    hidden_cosine_distance = float(torch.clamp(1.0 - cosine, 0.0, 2.0).item())

                token_ids.append(next_token_id)
                confidences.append(confidence)
                shifts.append(shift)
                token_margins.append(token_margin)
                token_entropies.append(token_entropy)
                hidden_cosine_distances.append(hidden_cosine_distance)
                hidden_norms.append(hidden_norm)
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
        if hidden_sum is None:
            mean_hidden = current_hidden.float().reshape(-1)
        else:
            mean_hidden = hidden_sum / len(token_ids)
        return GenerationTrace(
            answer=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            token_ids=token_ids,
            token_pieces=[
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids
            ],
            confidences=confidences,
            shifts=shifts,
            token_margins=token_margins,
            token_entropies=token_entropies,
            hidden_cosine_distances=hidden_cosine_distances,
            hidden_norms=hidden_norms,
            mean_hidden=[float(value) for value in mean_hidden.cpu().tolist()],
            elapsed_seconds=elapsed,
        )

    def score_candidate(
        self, knowledge: str, question: str, candidate_answer: str
    ) -> GenerationTrace:
        """Teacher-force one supplied grounded answer and capture prediction signals.

        At answer token ``t``, logits and hidden state come from the immediately
        preceding sequence position—the same alignment used during autoregressive
        generation. No candidate text is generated or altered.
        """
        torch = self.torch
        prompt_ids = self._grounded_prompt_ids(knowledge, question)
        answer_ids = self.tokenizer(
            candidate_answer.strip(),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        if answer_ids.shape[-1] == 0:
            raise ValueError("candidate answer must contain at least one token")

        prompt_length = int(prompt_ids.shape[-1])
        answer_length = int(answer_ids.shape[-1])
        full_ids = torch.cat([prompt_ids, answer_ids], dim=-1)
        start = prompt_length - 1
        end = start + answer_length

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=full_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits[:, start:end, :].float()
            hidden = outputs.hidden_states[self.layer][:, start:end, :].float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = torch.exp(log_probabilities)
            candidate_probabilities = torch.gather(
                probabilities, -1, answer_ids.unsqueeze(-1)
            ).squeeze(-1)
            top_probabilities = torch.topk(probabilities, k=2, dim=-1).values
            margins = top_probabilities[..., 0] - top_probabilities[..., 1]
            entropies = -(
                probabilities * log_probabilities
            ).sum(dim=-1) / math.log(logits.shape[-1])
            norms = torch.linalg.vector_norm(hidden, dim=-1) / math.sqrt(hidden.shape[-1])

            shifts: list[float | None] = [None]
            cosine_distances: list[float | None] = [None]
            if answer_length > 1:
                current = hidden[:, 1:, :]
                previous = hidden[:, :-1, :]
                shift_values = torch.linalg.vector_norm(
                    current - previous, dim=-1
                ) / (torch.linalg.vector_norm(previous, dim=-1) + 1e-8)
                cosine = torch.nn.functional.cosine_similarity(
                    current, previous, dim=-1, eps=1e-8
                )
                shifts.extend(float(value) for value in shift_values[0].tolist())
                cosine_distances.extend(
                    float(value)
                    for value in torch.clamp(1.0 - cosine, 0.0, 2.0)[0].tolist()
                )

        token_ids = [int(value) for value in answer_ids[0].tolist()]
        elapsed = time.perf_counter() - started
        mean_hidden = hidden.mean(dim=1)[0]
        return GenerationTrace(
            answer=candidate_answer.strip(),
            token_ids=token_ids,
            token_pieces=[
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids
            ],
            confidences=[float(value) for value in candidate_probabilities[0].tolist()],
            shifts=shifts,
            token_margins=[float(value) for value in margins[0].tolist()],
            token_entropies=[float(value) for value in entropies[0].tolist()],
            hidden_cosine_distances=cosine_distances,
            hidden_norms=[float(value) for value in norms[0].tolist()],
            mean_hidden=[float(value) for value in mean_hidden.cpu().tolist()],
            elapsed_seconds=elapsed,
        )
