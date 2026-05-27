from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import requests


@dataclass
class CheckRecord:
    index: int
    suite: str
    prediction: str
    expected: str
    expected_match: bool
    live_result: bool | None
    live_method: str
    live_reason: str
    endpoint_error: str
    is_correct: bool | None


SUITE_EXAMPLES: dict[str, list[tuple[str, str, bool]]] = {
    "exact": [
        ("parachute", "parachute", True),
        ("basketball", "basketball", True),
        ("soccer ball", "soccer ball", True),
        ("tennis ball", "tennis ball", True),
        ("baseball bat", "baseball bat", True),
        ("skateboard", "skateboard", True),
        ("surfboard", "surfboard", True),
        ("dog", "dog", True),
        ("cat", "cat", True),
        ("fish", "fish", True),
    ],
    "semantic_related": [
        ("irish terrier", "dog", True),
        ("tabby", "cat", True),
        ("persian cat", "cat", True),
        ("goldfish", "fish", True),
        ("alligator lizard", "reptile", True),
        ("snail", "mollusk", True),
        ("black and gold garden spider", "arachnid", True),
        ("american lobster", "crustacean", True),
        ("bullfrog", "amphibian", True),
        ("sorrel", "equine", True),
        ("wild boar", "porcine", True),
        ("parachute canopy", "parachute", True),
    ],
    "sports_objects": [
        ("football", "soccer ball", True),
        ("basket ball", "basketball", True),
        ("tennisball", "tennis ball", True),
        ("baseballbat", "baseball bat", True),
        ("airplane parachute", "parachute", True),
    ],
    "negatives": [
        ("street sign", "porcine", False),
        ("parachute", "basketball", False),
        ("soccer ball", "surfboard", False),
        ("tabby cat", "reptile", False),
        ("american lobster", "amphibian", False),
        ("snail", "arachnid", False),
        ("baseball bat", "tennis ball", False),
        ("parachute", "crustacean", False),
        ("basketball", "mollusk", False),
    ],
}


def _all_suite_names() -> list[str]:
    return sorted(SUITE_EXAMPLES.keys())


def _resolve_suites(raw_suites: list[str]) -> list[str]:
    if not raw_suites or "all" in raw_suites:
        return _all_suite_names()
    allowed = set(_all_suite_names())
    selected: list[str] = []
    for name in raw_suites:
        if name not in allowed:
            raise ValueError(f"Unknown suite '{name}'. Allowed: {', '.join(sorted(allowed))}, all")
        if name not in selected:
            selected.append(name)
    return selected


def _build_records(selected_suites: list[str], extra_examples: list[str]) -> list[CheckRecord]:
    examples: list[tuple[str, str, str, bool]] = []
    for suite in selected_suites:
        for prediction, expected, expected_match in SUITE_EXAMPLES[suite]:
            examples.append((suite, prediction, expected, expected_match))

    for raw in extra_examples:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) == 3:
            suite_name = "custom"
            prediction, expected, expected_match_raw = parts
        elif len(parts) == 4:
            suite_name, prediction, expected, expected_match_raw = parts
            suite_name = suite_name or "custom"
        else:
            raise ValueError(
                f"Invalid --example '{raw}'. Use prediction|target|expected_match or suite|prediction|target|expected_match"
            )
        expected_match = expected_match_raw.lower() in {"true", "1", "yes", "y"}
        examples.append((suite_name, prediction, expected, expected_match))

    return [
        CheckRecord(
            index=i + 1,
            suite=suite,
            prediction=prediction,
            expected=expected,
            expected_match=expected_match,
            live_result=None,
            live_method="",
            live_reason="",
            endpoint_error="",
            is_correct=None,
        )
        for i, (suite, prediction, expected, expected_match) in enumerate(examples)
    ]


def _parse_live_result(data: Any) -> tuple[bool | None, str, str]:
    if not isinstance(data, dict):
        return None, "", ""
    value = data.get("is_match")
    reason = str(data.get("reason", "") or "")
    method = str(data.get("method", "") or "")
    if isinstance(value, bool):
        return value, reason, method
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True, reason, method
        if lowered in {"false", "0", "no"}:
            return False, reason, method
    return None, reason, method


def _verify_live(
    records: list[CheckRecord],
    endpoint: str,
    llm_model: str,
    timeout_seconds: float,
) -> None:
    for rec in records:
        payload = {
            "prediction": rec.prediction,
            "target_label": rec.expected,
            "llm_model": llm_model,
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data: Any = response.json()
            parsed_result, parsed_reason, parsed_method = _parse_live_result(data)
            rec.live_result = parsed_result
            rec.live_reason = parsed_reason
            rec.live_method = parsed_method
            if rec.live_result is None:
                rec.endpoint_error = "missing/invalid is_match in response"
        except Exception as exc:
            rec.endpoint_error = str(exc)
        if rec.live_result is not None:
            rec.is_correct = rec.live_result == rec.expected_match


def _binary_metrics(records: list[CheckRecord]) -> dict[str, Any]:
    known = [r for r in records if r.live_result is not None]
    tp = sum(1 for r in known if r.expected_match and r.live_result is True)
    tn = sum(1 for r in known if (not r.expected_match) and r.live_result is False)
    fp = sum(1 for r in known if (not r.expected_match) and r.live_result is True)
    fn = sum(1 for r in known if r.expected_match and r.live_result is False)

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None
    accuracy = ((tp + tn) / len(known)) if known else None

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _suite_summary(records: list[CheckRecord]) -> dict[str, Any]:
    per_suite: dict[str, list[CheckRecord]] = {}
    for rec in records:
        per_suite.setdefault(rec.suite, []).append(rec)

    output: dict[str, Any] = {}
    for suite, items in sorted(per_suite.items()):
        known = [r for r in items if r.live_result is not None]
        correct = [r for r in known if r.is_correct is True]
        output[suite] = {
            "total": len(items),
            "known": len(known),
            "accuracy": (len(correct) / len(known)) if known else None,
            "errors": sum(1 for r in items if r.endpoint_error),
        }
    return output


def _method_breakdown(records: list[CheckRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        if rec.live_result is None:
            continue
        method = rec.live_method or "unknown"
        counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _summarize(records: list[CheckRecord]) -> dict[str, Any]:
    total = len(records)
    live_known = [r for r in records if r.live_result is not None]
    correct = [r for r in live_known if r.is_correct is True]

    return {
        "total_examples": total,
        "live_result_available": len(live_known),
        "accuracy": (len(correct) / len(live_known)) if live_known else None,
        "pass_count": len(correct),
        "fail_count": len(live_known) - len(correct),
        "endpoint_error_count": sum(1 for r in records if r.endpoint_error),
        "by_suite": _suite_summary(records),
        "method_breakdown": _method_breakdown(records),
        "binary_metrics": _binary_metrics(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check llm challenge verification accuracy using expanded suites."
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://127.0.0.1:8081/verify-label",
        help="Local LLM verification endpoint URL.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("PERTURB_LLM_ENDPOINT_MODEL", "Qwen2.5-1.5B-Instruct"),
        help="Model hint passed to verification endpoint.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout for each live verification request.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=["all"],
        help="Suite to run (can repeat): all, exact, semantic_related, sports_objects, negatives",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Add example: prediction|target|expected_match or suite|prediction|target|expected_match",
    )
    parser.add_argument(
        "--fail-only",
        action="store_true",
        help="Print only incorrect cases and endpoint errors in per-example output.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for full JSON report.",
    )
    args = parser.parse_args()

    parsed = urlparse(args.llm_endpoint)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost"}:
        print(
            f"llm endpoint must be local (127.0.0.1 or localhost), got: {args.llm_endpoint}",
            file=sys.stderr,
        )
        return 1

    try:
        selected_suites = _resolve_suites(args.suite)
        records = _build_records(selected_suites=selected_suites, extra_examples=args.example)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _verify_live(
        records=records,
        endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        timeout_seconds=args.timeout_seconds,
    )
    summary = _summarize(records)

    print("=== LLM Manual Challenge Accuracy Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSuites: {', '.join(selected_suites)}")
    print("\n=== Per-Example Results ===")
    for rec in records:
        if args.fail_only and not (rec.endpoint_error or rec.is_correct is False):
            continue
        print(
            f"[{rec.index}] suite={rec.suite} pred='{rec.prediction}' expected='{rec.expected}' expected_match={rec.expected_match} "
            f"live={rec.live_result} method='{rec.live_method}' correct={rec.is_correct} "
            f"error='{rec.endpoint_error}' reason='{rec.live_reason}'"
        )

    if args.output_json:
        payload = {
            "selected_suites": selected_suites,
            "summary": summary,
            "records": [asdict(r) for r in records],
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved full report to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

