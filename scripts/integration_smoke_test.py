from __future__ import annotations

import argparse
import base64
import sys
from typing import Any

import requests
import torch

from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_v2_l, predict_label


def _require_ok(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"{context} failed: {exc}") from exc
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"{context} returned non-object JSON")
    return data


def _verify(llm_endpoint: str, prediction: str, target: str, model: str) -> dict[str, Any]:
    payload = {"prediction": prediction, "target_label": target, "llm_model": model}
    response = requests.post(f"{llm_endpoint}/verify-label", json=payload, timeout=10)
    data = _require_ok(response, "llm_endpoint check")
    if "is_match" not in data:
        raise RuntimeError("llm_endpoint response missing is_match")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Perturb subnet local integration smoke test")
    parser.add_argument("--image-endpoint", default="https://api.pexels.com/v1/search")
    parser.add_argument("--pexels-api-key", default="")
    parser.add_argument("--pexels-image-variant", default="medium")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--label", default="dog")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--llm-model", default="qwen2.5:1.5b-instruct")
    args = parser.parse_args()

    print("[1/5] LLM endpoint health check")
    health = _require_ok(requests.get(f"{args.llm_endpoint}/health", timeout=5), "llm_endpoint health")
    print(f"  status={health.get('status')} default_model={health.get('default_model')}")

    print("[2/5] LLM endpoint semantic sanity checks")
    positive = _verify(args.llm_endpoint, prediction="irish terrier", target="dog", model=args.llm_model)
    negative = _verify(args.llm_endpoint, prediction="tabby cat", target="dog", model=args.llm_model)
    print(f"  positive_match={positive.get('is_match')} method={positive.get('method')}")
    print(f"  negative_match={negative.get('is_match')} method={negative.get('method')}")
    if not bool(positive.get("is_match")):
        raise RuntimeError("Expected positive semantic match for irish terrier vs dog")
    if bool(negative.get("is_match")):
        raise RuntimeError("Expected negative semantic match for tabby cat vs dog")

    print("[3/5] Fetch image challenge candidate")
    pexels_api_key = args.pexels_api_key.strip()
    if not pexels_api_key:
        raise RuntimeError("Missing --pexels-api-key")
    params = {
        "query": args.label,
        "page": 1,
        "per_page": 10,
    }
    image_data = _require_ok(
        requests.get(
            args.image_endpoint,
            params=params,
            headers={"Authorization": pexels_api_key},
            timeout=12,
        ),
        "pexels search",
    )
    photos = image_data.get("photos")
    if not isinstance(photos, list) or not photos:
        raise RuntimeError("pexels search returned no photos")
    src = photos[0].get("src", {}) if isinstance(photos[0], dict) else {}
    if not isinstance(src, dict):
        src = {}
    image_url = (
        src.get(args.pexels_image_variant)
        or src.get("medium")
        or src.get("large")
        or src.get("original")
    )
    if not isinstance(image_url, str) or not image_url.strip():
        raise RuntimeError("pexels photo src missing usable url")
    image_response = requests.get(image_url, timeout=12)
    image_response.raise_for_status()
    if not image_response.content:
        raise RuntimeError("downloaded pexels image is empty")
    image_b64 = base64.b64encode(image_response.content).decode("utf-8")

    print("[4/5] Run EfficientNetV2-L inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device=device)
    image = decode_image_b64(image_b64).to(device)
    prediction = predict_label(model=model, image_chw=image)
    print(f"  model_prediction={prediction}")

    print("[5/5] Verify challenge label semantics through local llm_endpoint")
    challenge_check = _verify(
        args.llm_endpoint,
        prediction=prediction,
        target=args.label,
        model=args.llm_model,
    )
    print(
        "  challenge_is_match="
        f"{challenge_check.get('is_match')} method={challenge_check.get('method')} reason={challenge_check.get('reason')}"
    )
    if not bool(challenge_check.get("is_match")):
        raise RuntimeError("Challenge candidate did not pass semantic verification")

    metrics = _require_ok(requests.get(f"{args.llm_endpoint}/metrics", timeout=5), "metrics check")
    print(
        "  llm_endpoint_metrics="
        f"total={metrics.get('total_requests')} llm_failures={metrics.get('llm_failures')}"
    )
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
