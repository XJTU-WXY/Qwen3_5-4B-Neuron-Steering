"""MLP neuron steering for Qwen3.5-4B.

A push on neuron n in layer L adds

    alpha = dose * median(||h_L||, tokens 1:) / ||W_down[:, n]||

to that neuron's down_proj input at every token. down_proj has no bias, so this
is the same as adding alpha * W_down[:, n] to the MLP output.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Iterator

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

CALIBRATION_QUESTIONS = [
    "What is the capital of France?",
    "What does 17 times 24 equal?",
    "Explain in one sentence why the sky looks blue.",
    "Name three primary colors.",
    "Summarize what photosynthesis does in one sentence.",
    "Give one tip for sleeping better.",
]


class _CountStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.n_tokens = 0

    def put(self, value):
        try:
            self.n_tokens += int(value.numel())
        except AttributeError:
            pass
        super().put(value)


class _StopOnFlag(StoppingCriteria):
    def __init__(self, flag: threading.Event):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.flag.is_set()


class Steerer:
    def __init__(self, model_path: str, device: str = "cuda:0", dtype=None,
                 calibration_path: str | None = None, verbose: bool = True):
        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.dtype = dtype or (torch.float16 if device.startswith("cuda") else torch.float32)
        self.calibration_path = calibration_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "calibration.json"
        )
        self.verbose = verbose

        self.tokenizer = None
        self.model = None
        self.lm = None
        self.num_layers = 0
        self.num_neurons = 0
        self.hidden_size = 0
        self.layer_types: list[str] = []

        self.medians: list[float] = []
        self.calibration_meta: dict = {}
        self._column_norms: dict[tuple[int, int], float] = {}
        self._hooks = []
        self._delta_norm: list[float] = []

        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._loaded_at = 0.0

    # ------------------------------------------------------------------ load

    def load(self):
        t0 = time.time()
        self._log(f"loading tokenizer {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._log(f"loading weights ({self.dtype})")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path, dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        self.lm = self.model.model.language_model
        text_cfg = self.model.config.text_config
        self.num_layers = text_cfg.num_hidden_layers
        self.num_neurons = text_cfg.intermediate_size
        self.hidden_size = text_cfg.hidden_size
        self.layer_types = list(text_cfg.layer_types)
        self._loaded_at = time.time() - t0
        self._log(f"model ready in {self._loaded_at:.1f}s")

    def calibrate(self, force: bool = False) -> None:
        cached = self._read_calibration()
        if cached is not None and not force:
            self.medians = cached["medians"]
            self.calibration_meta = cached
            self._log(f"calibration loaded (median L0..L{self.num_layers - 1}: "
                      f"{self.medians[0]:.2f} .. {self.medians[-1]:.2f})")
            return

        t0 = time.time()
        self._log("calibrating residual norms on six plain answers")
        per_layer: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        answers = []
        total_tokens = 0
        recorded: dict[int, torch.Tensor] = {}
        handles = []

        for index, layer in enumerate(self.lm.layers):
            def capture(module, args, output, index=index):
                recorded[index] = output[0] if isinstance(output, tuple) else output
                return None
            handles.append(layer.register_forward_hook(capture))

        try:
            for question in CALIBRATION_QUESTIONS:
                input_ids, attention_mask = self.build_inputs(
                    [{"role": "user", "content": question}], thinking=False
                )
                with torch.inference_mode():
                    ids = self.model.generate(
                        input_ids, attention_mask=attention_mask, max_new_tokens=40,
                        do_sample=False, cache_implementation="dynamic",
                    )
                answers.append(self.tokenizer.decode(ids[0, input_ids.shape[1]:],
                                                     skip_special_tokens=True).strip())
                with torch.inference_mode():
                    self.lm(input_ids=ids)
                for layer in range(self.num_layers):
                    h = recorded[layer][0].float()
                    if h.shape[0] > 1:
                        per_layer[layer].append(h[1:].norm(dim=-1))
                total_tokens += int(ids.shape[1])
                recorded.clear()
        finally:
            for handle in handles:
                handle.remove()

        self.medians = [float(torch.cat(v).median()) for v in per_layer]
        self.calibration_meta = {
            "model": self.model_path,
            "dtype": str(self.dtype).replace("torch.", ""),
            "questions": CALIBRATION_QUESTIONS,
            "answers": answers,
            "tokens": total_tokens,
            "seconds": round(time.time() - t0, 1),
            "medians": self.medians,
        }
        with open(self.calibration_path, "w", encoding="utf-8") as fh:
            json.dump(self.calibration_meta, fh, indent=2, ensure_ascii=False)
        self._log(f"calibration written to {self.calibration_path}")

    def _read_calibration(self):
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("model") != self.model_path:
            return None
        if len(data.get("medians", [])) != self.num_layers:
            return None
        if data.get("dtype") != str(self.dtype).replace("torch.", ""):
            return None
        return data

    # --------------------------------------------------------------- steering

    def column_norm(self, layer: int, neuron: int) -> float:
        key = (layer, neuron)
        if key not in self._column_norms:
            weight = self.lm.layers[layer].mlp.down_proj.weight
            self._column_norms[key] = float(weight[:, neuron].float().norm())
        return self._column_norms[key]

    def alpha(self, layer: int, neuron: int, dose: float) -> float:
        return dose * self.medians[layer] / self.column_norm(layer, neuron)

    def clear(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._delta_norm = []

    def describe_recipe(self, recipe) -> list[dict]:
        detail = []
        for layer, neuron, dose in recipe:
            layer, neuron, dose = int(layer), int(neuron), float(dose)
            if not 0 <= layer < self.num_layers:
                raise ValueError(f"layer {layer} out of range 0..{self.num_layers - 1}")
            if not 0 <= neuron < self.num_neurons:
                raise ValueError(f"neuron {neuron} out of range 0..{self.num_neurons - 1}")
            detail.append({
                "layer": layer,
                "neuron": neuron,
                "dose": dose,
                "alpha": self.alpha(layer, neuron, dose),
                "column_norm": self.column_norm(layer, neuron),
                "residual_median": self.medians[layer],
            })
        return detail

    def apply(self, recipe) -> list[dict]:
        """recipe: iterable of (layer, neuron, dose). Returns per-neuron detail."""
        self.clear()
        detail = [item for item in self.describe_recipe(recipe) if item["dose"] != 0]
        grouped: dict[int, list[tuple[int, float]]] = {}
        for item in detail:
            grouped.setdefault(item["layer"], []).append((item["neuron"], item["alpha"]))

        for layer, pairs in sorted(grouped.items()):
            delta = torch.zeros(self.num_neurons, device=self.device, dtype=torch.float32)
            for neuron, value in pairs:
                delta[neuron] += value
            delta = delta.to(self.dtype)
            self._delta_norm.append(float(delta.float().norm()))

            def hook(module, args, delta=delta):
                return (args[0] + delta.to(args[0].dtype),) + tuple(args[1:])

            self._hooks.append(self.lm.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        return detail

    # -------------------------------------------------------------- inference

    def build_inputs(self, messages, thinking: bool):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def stop(self) -> None:
        self._stop.set()

    def stream(self, messages, recipe, max_new_tokens=320, temperature=0.7, top_p=0.9,
               repetition_penalty=1.05, thinking=False, label="single") -> Iterator[dict]:
        with self.lock:
            self._stop.clear()
            detail = self.apply(recipe or [])
            input_ids, attention_mask = self.build_inputs(messages, thinking)
            streamer = _CountStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
            )
            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                streamer=streamer,
                cache_implementation="dynamic",
                repetition_penalty=float(repetition_penalty),
                stopping_criteria=StoppingCriteriaList([_StopOnFlag(self._stop)]),
            )
            if float(temperature) > 0:
                kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
            else:
                kwargs.update(do_sample=False)

            yield {
                "type": "start",
                "label": label,
                "prompt_tokens": int(input_ids.shape[1]),
                "steering": detail,
            }
            worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            t0 = time.time()
            worker.start()
            try:
                for chunk in streamer:
                    if chunk:
                        yield {"type": "delta", "label": label, "text": chunk}
            finally:
                worker.join(timeout=2.0)
                self.clear()
            elapsed = time.time() - t0
            tokens = streamer.n_tokens
            yield {
                "type": "end",
                "label": label,
                "tokens": tokens,
                "seconds": round(elapsed, 2),
                "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
                "stopped": self._stop.is_set(),
            }

    def describe(self) -> dict:
        return {
            "model": os.path.basename(self.model_path),
            "path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "num_layers": self.num_layers,
            "num_neurons": self.num_neurons,
            "hidden_size": self.hidden_size,
            "layer_types": self.layer_types,
            "medians": self.medians,
            "calibration": {k: v for k, v in self.calibration_meta.items()
                            if k in ("tokens", "seconds", "questions")},
            "load_seconds": round(self._loaded_at, 1),
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[steer] {message}", flush=True)
