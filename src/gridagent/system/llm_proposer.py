"""LLM proposer front-end for the agentic pipeline.

The LLM reads the operator report + gathered evidence + the candidate action slate and
returns a preference ranking (best -> worst) with a short rationale. EA-VOI then decides
which of the LLM's proposals to physics-verify under budget and can override an unsafe
top pick. This is the "LLM proposes and explains; the allocator decides what to check"
design of docs/METHOD_SPEC.md. Local open-model inference (no proxy token available).
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any



def action_type(action: dict[str, Any]) -> str:
    return str(action.get("action_type") or action.get("type") or "")

_SYS = (
    "You are a distribution-grid control-room assistant. Given the operator report, the current "
    "grid evidence, and a numbered list of candidate corrective actions, RANK the actions from best "
    "to worst for safely and cost-effectively resolving the situation. Prefer actions that clear "
    "violations without introducing new ones; escalate or do nothing only when appropriate. "
    'Respond with ONLY a JSON object: {"ranking": ["a_XXX", ...], "rationale": "<one sentence>"}. '
    "The ranking MUST list every action id exactly once, best first."
)


def _ctx_summary(ctx: dict[str, Any]) -> str:
    bits = []
    if ctx.get("voltage_violation"):
        bits.append("voltage violation present")
    if ctx.get("loading_violation"):
        bits.append("line overload present")
    if ctx.get("failure_active"):
        bits.append("active failure alarm")
    return ", ".join(bits) or "no active violations detected"


def build_prompt(public_case: dict[str, Any], ctx: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    lines = [
        f"OPERATOR REPORT: {public_case.get('operator_report', '')}",
        f"TRIGGER: {public_case.get('trigger_type', 'unknown')}",
        f"GRID EVIDENCE: {_ctx_summary(ctx)}",
        "CANDIDATE ACTIONS:",
    ]
    for a in actions:
        lines.append(
            f"  {a['action_id']}: type={action_type(a)}, coarse_risk={a.get('coarse_risk', 'unknown')}, "
            f"coarse_cost={a.get('coarse_cost', 'unknown')} — {str(a.get('description', ''))[:140]}"
        )
    return "\n".join(lines)


def parse_ranking(text: str, valid_ids: list[str]) -> tuple[list[str], str]:
    """Extract a full ranking of valid action ids from the model output; robust to noise."""
    rationale = ""
    ranked: list[str] = []
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            rationale = str(obj.get("rationale", ""))[:200]
            ranked = [str(x) for x in (obj.get("ranking") or [])]
    except Exception:
        pass
    if not ranked:  # fallback: scrape a_### tokens in order of appearance
        ranked = re.findall(r"a_\d{3}", text)
    seen = set()
    valid = set(valid_ids)
    out = [i for i in ranked if i in valid and not (i in seen or seen.add(i))]
    # append any missing valid ids (stable) so the ranking is complete
    out += [i for i in valid_ids if i not in seen]
    return out, rationale


def _set_submodule_compat(self, target: str, module, strict: bool = False):
    """Backport torch.nn.Module.set_submodule for the Mistral3 loader."""
    if not target:
        raise ValueError("target must not be empty")
    if "." in target:
        parent_name, child_name = target.rsplit(".", 1)
        try:
            parent = self.get_submodule(parent_name)
        except AttributeError:
            if strict:
                raise
            parent = self
            for atom in parent_name.split("."):
                if not hasattr(parent, atom):
                    raise AttributeError(f"parent module {parent_name!r} does not exist")
                parent = getattr(parent, atom)
    else:
        parent, child_name = self, target
    if strict and not hasattr(parent, child_name):
        raise AttributeError(f"submodule {target!r} does not exist")
    setattr(parent, child_name, module)


@contextmanager
def module_set_submodule_compat():
    """Install the backport only while Transformers constructs the model."""
    import torch

    module_class = torch.nn.Module
    if hasattr(module_class, "set_submodule"):
        yield False
        return
    setattr(module_class, "set_submodule", _set_submodule_compat)
    try:
        yield True
    finally:
        delattr(module_class, "set_submodule")


class LocalLLM:
    """Load one local HF instruct model with an explicit decoding contract."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        decoding_seed: int = 0,
        revision: str | None = None,
    ) -> None:
        import hashlib
        import os
        import torch  # noqa
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.decoding_seed = int(decoding_seed)
        self.revision = revision
        self._generation_index = 0
        self.last_completion_tokens = 0   # real count of tokens generated on the last chat()
        hf_kwargs = {"trust_remote_code": True}
        if revision:
            hf_kwargs["revision"] = revision
        # Passing the token explicitly avoids an accidental unauthenticated fallback while
        # keeping it entirely in process memory; it is never copied to runtime metadata.
        token = os.environ.get("HF_TOKEN")
        if token:
            hf_kwargs["token"] = token
        try:  # some Mistral tokenizers need this flag for correct tokenization
            self.tok = AutoTokenizer.from_pretrained(
                model_id, fix_mistral_regex=True, **hf_kwargs
            )
        except (TypeError, ValueError):
            self.tok = AutoTokenizer.from_pretrained(model_id, **hf_kwargs)
        config = AutoConfig.from_pretrained(model_id, **hf_kwargs)
        self.model_type = str(getattr(config, "model_type", ""))
        architectures = tuple(getattr(config, "architectures", None) or ())
        # Gemma 4 (gemma4_unified) is a conditional-generation model even for text-only
        # prompts.  Mistral 3 and Gemma 3 use the same multimodal-compatible auto class.
        multimodal_types = {"mistral3", "gemma3", "gemma3_text", "gemma4_unified"}
        model_class = (
            AutoModelForImageTextToText
            if self.model_type in multimodal_types
            or any("ForConditionalGeneration" in a for a in architectures)
            else AutoModelForCausalLM
        )
        with module_set_submodule_compat() as used_compat:
            self.model = model_class.from_pretrained(
                model_id, torch_dtype="auto", device_map=device, **hf_kwargs
            )
        self.used_set_submodule_compat = bool(used_compat)
        self.loader_class = model_class.__name__
        self.device = device
        self.resolved_revision = str(
            revision
            or getattr(config, "_commit_hash", None)
            or getattr(self.model, "_commit_hash", None)
            or "unknown"
        )
        template = getattr(self.tok, "chat_template", None)
        if isinstance(template, dict):
            template = json.dumps(template, sort_keys=True, separators=(",", ":"))
        self.chat_template_sha256 = hashlib.sha256(
            str(template or "").encode("utf-8")
        ).hexdigest()

    def set_decoding_seed(self, seed: int) -> None:
        self.decoding_seed = int(seed)
        self._generation_index = 0

    def chat(self, system: str, user: str) -> str:
        return self.chat_messages([{"role": "system", "content": system},
                                   {"role": "user", "content": user}])

    def chat_messages(self, messages: list) -> str:
        """Multi-turn generation over a full message history (enables trajectory memory)."""
        import torch

        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        n_in = enc["input_ids"].shape[1]
        sampling = self.temperature > 0.0
        call_seed = self.decoding_seed + self._generation_index
        self._generation_index += 1
        if sampling:
            torch.manual_seed(call_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(call_seed)
        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": sampling,
            "pad_token_id": self.tok.eos_token_id,
        }
        if sampling:
            generation.update(temperature=self.temperature, top_p=self.top_p)
        with torch.no_grad():
            out = self.model.generate(**enc, **generation)
        gen = out[0][n_in:]
        self.last_completion_tokens = int(gen.shape[0])   # REAL generated-token count (gate-checked)
        return self.tok.decode(gen, skip_special_tokens=True)

    def propose(self, public_case: dict[str, Any], ctx: dict[str, Any], actions: list[dict[str, Any]]) -> tuple[list[str], str]:
        prompt = build_prompt(public_case, ctx, actions)
        text = self.chat(_SYS, prompt)
        return parse_ranking(text, [str(a["action_id"]) for a in actions])
