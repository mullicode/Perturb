from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9 ]")


def _normalize(text: str) -> str:
    raw = text.strip().lower().replace("_", " ").replace("-", " ")
    raw = _NON_ALNUM_SPACE_RE.sub(" ", raw)
    return _WHITESPACE_RE.sub(" ", raw).strip()


def _deterministic_match(prediction: str, target: str) -> tuple[bool, str] | None:
    pred_norm = _normalize(prediction)
    target_norm = _normalize(target)

    # Keep deterministic logic intentionally minimal; most decisions go to LLM.
    if pred_norm == target_norm:
        return True, "exact canonical match"
    return None


class VerifyRequest(BaseModel):
    prediction: str = Field(..., min_length=1)
    target_label: str = Field(..., min_length=1)
    llm_model: str | None = None


class VerifyResponse(BaseModel):
    is_match: bool
    reason: str
    method: str


@dataclass
class Metrics:
    started_at: float
    total_requests: int = 0
    deterministic_matches: int = 0
    llm_requests: int = 0
    llm_vote_rounds: int = 0
    llm_failures: int = 0


app = FastAPI(title="Perturb LLM Endpoint", version="0.1.0")
_metrics = Metrics(started_at=time.time())
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("PERTURB_LLM_ENDPOINT_MODEL", os.getenv("PERTURB_LLM_VERIFY_MODEL", "qwen2.5:1.5b-instruct"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "8"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.1"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "20"))
OLLAMA_VOTE_COUNT = max(1, int(os.getenv("OLLAMA_VOTE_COUNT", "3")))
OLLAMA_VOTE_TEMPERATURE = float(os.getenv("OLLAMA_VOTE_TEMPERATURE", "0.2"))
_RELATION_TRUE = {"same_or_near", "related"}
_RELATION_FALSE = {"different"}
_RELATION_VALUES = _RELATION_TRUE | _RELATION_FALSE | {"unknown"}


def _resolve_model_name(raw: str) -> str:
    value = raw.strip()
    lowered = value.lower()
    aliases = {
        "qwen2.5-1.5b-instruct": "qwen2.5:1.5b-instruct",
        "qwen2.5:1.5b-instruct": "qwen2.5:1.5b-instruct",
    }
    return aliases.get(lowered, value)


def _prompt(prediction: str, target: str, style: int) -> str:
    if style % 2 == 0:
        return (
            "You are a semantic similarity judge for image labels.\n"
            "Decide relation between prediction and target_label.\n"
            "Valid relation values: same_or_near, related, different, unknown.\n"
            "Rules:\n"
            "- same_or_near => labels are equivalent or near-synonyms.\n"
            "- related => labels are semantically related enough to be considered close.\n"
            "- different => labels are clearly different/unrelated in meaning.\n"
            "- unknown => only if unsure.\n"
            "Set is_match=true when relation is same_or_near or related.\n"
            "Set is_match=false when relation is different.\n"
            "If relation is unknown, choose the most likely relation and keep reason brief.\n"
            "Return ONLY JSON:\n"
            "{\"relation\":\"same_or_near|related|different|unknown\",\"is_match\":true|false,"
            "\"confidence\":0.0-1.0,\"reason\":\"short\"}\n"
            f"prediction={prediction}\n"
            f"target_label={target}\n"
        )
    return (
        "Classify semantic closeness between prediction and target_label.\n"
        "Return true when they are close enough in meaning, false when clearly different.\n"
        "Return ONLY JSON:\n"
        "{\"relation\":\"same_or_near|related|different|unknown\",\"is_match\":true|false,"
        "\"confidence\":0.0-1.0,\"reason\":\"short\"}\n"
        f"prediction={prediction}\n"
        f"target_label={target}\n"
    )


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(max(0.0, min(1.0, value)))
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return float(max(0.0, min(1.0, parsed)))
    return None


def _parse_llm_decision(raw: Any) -> tuple[bool, str]:
    if not isinstance(raw, dict):
        raise ValueError("LLM output must be JSON object")
    relation_raw = str(raw.get("relation", "") or "").strip().lower()
    relation = relation_raw if relation_raw in _RELATION_VALUES else ""
    parsed_bool = _coerce_bool(raw.get("is_match"))
    reason = str(raw.get("reason", "llm semantic decision") or "llm semantic decision")
    confidence = _coerce_confidence(raw.get("confidence"))

    relation_implied: bool | None = None
    if relation in _RELATION_TRUE:
        relation_implied = True
    elif relation in _RELATION_FALSE:
        relation_implied = False

    if relation_implied is None and parsed_bool is None:
        raise ValueError("LLM JSON missing usable relation/is_match")
    final_decision = parsed_bool if parsed_bool is not None else relation_implied
    if final_decision is None:
        raise ValueError("Unable to resolve final decision")
    if relation_implied is not None and parsed_bool is not None and relation_implied != parsed_bool:
        # Resolve inconsistent payloads by trusting explicit relation label.
        final_decision = relation_implied
        reason = f"{reason} (resolved via relation={relation})"
    if confidence is not None:
        reason = f"{reason} [confidence={confidence:.2f}]"
    return bool(final_decision), reason


def _ollama_match_once(prediction: str, target_label: str, model: str, style: int) -> tuple[bool, str]:
    resolved_model = _resolve_model_name(model)
    payload = {
        "model": resolved_model,
        "prompt": _prompt(prediction=prediction, target=target_label, style=style),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": OLLAMA_VOTE_TEMPERATURE if OLLAMA_VOTE_COUNT > 1 else OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    raw = body.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Invalid Ollama response payload")
    parsed: Any = json.loads(raw)
    return _parse_llm_decision(parsed)


def _majority_threshold(votes: int) -> int:
    return (votes // 2) + 1


def _ollama_match(prediction: str, target_label: str, model: str) -> tuple[bool, str]:
    votes = max(1, OLLAMA_VOTE_COUNT)
    positives = 0
    negatives = 0
    reasons: list[str] = []
    errors: list[str] = []

    for idx in range(votes):
        _metrics.llm_vote_rounds += 1
        try:
            decision, reason = _ollama_match_once(prediction=prediction, target_label=target_label, model=model, style=idx)
            reasons.append(reason)
            if decision:
                positives += 1
            else:
                negatives += 1
        except Exception as exc:
            errors.append(str(exc))

    valid_votes = positives + negatives
    if valid_votes == 0:
        raise ValueError(f"All LLM votes failed: {errors}")

    threshold = _majority_threshold(valid_votes)
    final = positives >= threshold
    winner = "true" if final else "false"
    # Keep one representative reason from the winning side.
    representative_reason = reasons[0] if reasons else "llm vote decision"
    reason = (
        f"vote={winner} positives={positives} negatives={negatives} "
        f"valid_votes={valid_votes}/{votes}; {representative_reason}"
    )
    if errors:
        reason = f"{reason}; vote_errors={len(errors)}"
    return final, reason


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - _metrics.started_at),
        "default_model": DEFAULT_MODEL,
        "ollama_url": OLLAMA_URL,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "uptime_seconds": int(time.time() - _metrics.started_at),
        "total_requests": _metrics.total_requests,
        "deterministic_matches": _metrics.deterministic_matches,
        "llm_requests": _metrics.llm_requests,
        "llm_vote_rounds": _metrics.llm_vote_rounds,
        "llm_failures": _metrics.llm_failures,
    }


@app.post("/verify-label", response_model=VerifyResponse)
def verify_label(req: VerifyRequest) -> VerifyResponse:
    _metrics.total_requests += 1
    prediction = _normalize(req.prediction)
    target = _normalize(req.target_label)
    model = _resolve_model_name((req.llm_model or DEFAULT_MODEL).strip())
    if not prediction or not target:
        raise HTTPException(status_code=400, detail="prediction and target_label are required")

    deterministic = _deterministic_match(prediction=prediction, target=target)
    if deterministic is not None:
        _metrics.deterministic_matches += 1
        decision, reason = deterministic
        return VerifyResponse(is_match=decision, reason=reason, method="deterministic")

    _metrics.llm_requests += 1
    try:
        is_match, reason = _ollama_match(prediction=prediction, target_label=target, model=model)
        method = "ollama_vote" if OLLAMA_VOTE_COUNT > 1 else "ollama"
        return VerifyResponse(is_match=is_match, reason=reason, method=method)
    except Exception as exc:
        _metrics.llm_failures += 1
        raise HTTPException(status_code=502, detail=f"llm endpoint failed: {exc}") from exc


# Backward-compatible alias for prior name.
@app.post("/match-label", response_model=VerifyResponse)
def match_label_alias(req: VerifyRequest) -> VerifyResponse:
    return verify_label(req)
