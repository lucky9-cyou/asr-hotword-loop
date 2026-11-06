"""Local Qwen3-ASR decoding with prefix-conditioned logit bias and real traces.

Each transcribe call is one independent audio segment. The caller owns streaming
segmentation and passes the latest hotword snapshot for every new segment.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

DEPS = Path(__file__).resolve().parent / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))


class HotwordLogitsProcessor:
    """Apply a phrase-prefix bonus, recording before/after at the same prefix.

    Scores are model logits after any standard generation processors. Candidate
    probabilities use the full vocabulary denominator (not a top-k renormalization).
    The trace is conditional on the biased run's prefix, not a sequence likelihood.
    """

    def __init__(self, tokenizer, hotwords, bias, prompt_length=0):
        self.tokenizer = tokenizer
        self.words = list(dict.fromkeys(w.strip() for w in hotwords if isinstance(w, str) and w.strip()))[:64]
        self.phrases = [(w, tokenizer.encode(w, add_special_tokens=False)) for w in self.words]
        self.bias = max(0.0, min(12.0, float(bias)))
        self.prompt_length = prompt_length
        self.records = []

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids[0, self.prompt_length:].tolist()
        prefix = self.tokenizer.decode(generated, skip_special_tokens=True)
        raw = scores[0].float()
        best = float(raw.max())
        bonuses = {}
        reasons = {}
        for word, ids in self.phrases:
            if not ids:
                continue
            # Longest matching suffix of generated tokens against phrase prefix.
            matched = 0
            for n in range(min(len(ids) - 1, len(generated)), 0, -1):
                if generated[-n:] == ids[:n]:
                    matched = n
                    break
            target = ids[matched]
            # A rare name can begin in a BPE token merged with preceding text.
            # Matching decoded characters covers that alternative token boundary.
            if not matched:
                for n in range(len(word) - 1, 0, -1):
                    if prefix.endswith(word[:n]):
                        remaining = self.tokenizer.encode(word[n:], add_special_tokens=False)
                        if remaining:
                            target = remaining[0]
                            matched = n
                        break
            bonus = self.bias if matched else self.bias * 0.2
            # Weak starts need acoustic support; continuations allow rare spellings.
            gate = 16.0 if matched else 4.0
            if best - float(raw[target]) <= gate and bonus > 0:
                if bonus > bonuses.get(target, 0):
                    bonuses[target] = bonus
                    reasons[target] = {"word": word, "matched_prefix": bool(matched)}

        adjusted = scores.clone()
        for tid, bonus in bonuses.items():
            adjusted[0, tid] += bonus
        before = torch.softmax(raw, dim=-1)
        after = torch.softmax(adjusted[0].float(), dim=-1)
        # Include competing high-probability tokens as well as every boosted token.
        ids = set(torch.topk(before, min(5, before.numel())).indices.tolist())
        ids.update(torch.topk(after, min(5, after.numel())).indices.tolist())
        ids.update(bonuses)
        candidates = []
        for tid in sorted(ids, key=lambda t: float(after[t]), reverse=True):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            full_text = self.tokenizer.decode(generated + [tid], skip_special_tokens=True)
            if full_text.startswith(prefix) and full_text[len(prefix):]:
                text = full_text[len(prefix):]
            candidates.append({
                "token_id": tid, "token": text,
                "token_piece": self.tokenizer.convert_ids_to_tokens(tid),
                "prob_before": float(before[tid]), "prob_after": float(after[tid]),
                "logit_before": float(raw[tid]),
                "logit_after": float(adjusted[0, tid]),
                "boost": bonuses.get(tid, 0.0),
                **reasons.get(tid, {}),
            })
        self.records.append({
            "step": len(generated), "prefix": prefix,
            "selected_token_id": int(adjusted[0].argmax()),
            "candidates": candidates,
            "active": bool(bonuses),
            "probability_kind": "full_vocabulary_softmax_same_prefix",
        })
        return adjusted


class ASREngine:
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.getenv("ASR_MODEL", "/workspace/model/Qwen3-ASR-0.6B")
        self.device_requested = device or os.getenv("ASR_DEVICE", "auto")
        self._lock = threading.RLock()
        self.model = None
        self.processor = None
        self.wrapper = None
        self.device = None
        self.error = None

    def status(self):
        return {"model": self.model_path, "loaded": self.model is not None,
                "device": self.device or self.device_requested, "error": self.error}

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            try:
                import torch
                from qwen_asr import Qwen3ASRModel

                torch.set_num_threads(int(os.getenv("ASR_CPU_THREADS", "4")))
                device = self.device_requested
                if device == "auto":
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.device = device
                dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
                self.wrapper = Qwen3ASRModel.from_pretrained(
                    self.model_path, dtype=dtype, device_map=device,
                    local_files_only=True, attn_implementation="sdpa",
                    max_inference_batch_size=1, max_new_tokens=160,
                )
                self.model = self.wrapper.model.eval()
                self.processor = self.wrapper.processor
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                raise

    @staticmethod
    def read_audio(source):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        if isinstance(source, tuple):
            audio, rate = source
        else:
            audio, rate = sf.read(source, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Audio must contain a non-empty mono or stereo waveform")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples")
        rate = int(rate)
        if rate < 1000 or rate > 384000:
            raise ValueError("Unsupported audio sample rate")
        if rate != 16000:
            common = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // common, rate // common).astype(np.float32)
        if len(audio) > 16000 * 30:
            raise ValueError("One ASR segment must be at most 30 seconds")
        return audio

    def transcribe(self, audio_path, hotwords=None, bias=8.0):
        import numpy as np
        import torch
        from transformers import LogitsProcessorList

        started = time.monotonic()
        wave = self.read_audio(audio_path)
        duration = len(wave) / 16000
        hotwords = hotwords or []
        with self._lock, torch.inference_mode():
            self.load()
            if float(np.sqrt(np.mean(wave ** 2))) < 1e-5:
                return {"baseline": "", "biased": "", "probabilities": [],
                        "model": self.model_path, "device": self.device,
                        "elapsed_s": time.monotonic() - started,
                        "audio_seconds": duration, "silence": True}
            prompt = self.wrapper._build_text_prompt(context="", force_language="Chinese")
            inputs = self.processor(text=[prompt], audio=[wave], return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            prompt_length = inputs["input_ids"].shape[1]
            max_tokens = min(192, max(40, int(duration * 12) + 20))

            def run(words, strength):
                tracer = HotwordLogitsProcessor(self.processor.tokenizer, words, strength, prompt_length)
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    logits_processor=LogitsProcessorList([tracer]),
                )
                text = self.processor.batch_decode(
                    out.sequences[:, prompt_length:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                return text, tracer.records

            baseline, baseline_trace = run([], 0)
            if hotwords and float(bias) > 0:
                biased, trace = run(hotwords, bias)
            else:
                biased, trace = baseline, baseline_trace
            return {
                "baseline": baseline, "biased": biased, "probabilities": trace,
                "baseline_probabilities": baseline_trace,
                "model": self.model_path, "device": self.device,
                "elapsed_s": round(time.monotonic() - started, 3),
                "audio_seconds": round(duration, 3), "bias": float(bias),
                "hotwords_applied": list(hotwords),
                "probability_note": "真实 Qwen3-ASR logits；同一音频和偏置解码前缀下，完整词表 softmax 偏置前/后概率。不是整词置信度。",
            }
