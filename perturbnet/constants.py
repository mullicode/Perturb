from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default

# Shared subnet identity/constants.
SUBNET_NAMESPACE = "perturb"
MODEL_NAME = "EfficientNetV2-M"
PROMPTS = (
    "dog",
    "cat",
    "bird",
    "fish",
    "reptile",
    "amphibian",
    "insect",
    "arachnid",
    "crustacean",
    "mollusk",
    "primate",
    "rodent",
    "rabbit",
    "equine",
    "bovine",
    "ovine",
    "caprine",
    "porcine",
    "ungulate",
    "marine_mammal",
)

# Validator runtime state files.
VALIDATOR_STATE_FILENAME = "perturb_validator_state.json"
FALLBACK_IMAGE_RELATIVE_PATH = "assets/dog_1.jpg"
FALLBACK_LABEL = "dog"

# Validator runtime constants.
IMAGE_ENDPOINT = os.getenv("PERTURB_IMAGE_ENDPOINT", "https://api.pexels.com/v1/search")
PEXELS_API_KEY = _env_first(("PERTURB_PEXELS_API_KEY", "PEXELS_API_KEY"), "")
PEXELS_PER_PAGE = _env_int("PERTURB_PEXELS_PER_PAGE", 40)
PEXELS_PAGE_SPAN = _env_int("PERTURB_PEXELS_PAGE_SPAN", 10)
PEXELS_IMAGE_VARIANT = os.getenv("PERTURB_PEXELS_IMAGE_VARIANT", "medium")
IMAGE_SIZE = _env_int("PERTURB_IMAGE_SIZE", 64)
TIMEOUT_SECONDS = _env_int("PERTURB_TIMEOUT_SECONDS", 30)
QUERY_INTERVAL_SECONDS = _env_int("PERTURB_QUERY_INTERVAL_SECONDS", 120)
K_MINERS = _env_int("PERTURB_K_MINERS", 100)
HISTORY_SIZE = _env_int("PERTURB_HISTORY_SIZE", 50)
MIN_PROCESSED_COUNT = _env_int("PERTURB_MIN_PROCESSED_COUNT", 50)
LLM_ENDPOINT_URL = _env_first(
    ("PERTURB_LLM_ENDPOINT_URL", "PERTURB_LABEL_MATCH_ENDPOINT", "PERTURB_LLM_VERIFY_ENDPOINT"),
    "http://127.0.0.1:8081/verify-label",
)
LLM_ENDPOINT_MODEL = _env_first(
    ("PERTURB_LLM_ENDPOINT_MODEL", "PERTURB_LABEL_MATCH_MODEL", "PERTURB_LLM_VERIFY_MODEL"),
    "Qwen2.5-1.5B-Instruct",
)
LLM_ENDPOINT_TIMEOUT_SECONDS = _env_int("PERTURB_LLM_ENDPOINT_TIMEOUT_SECONDS", 20)
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.003)
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
LINF_COMPONENT_WEIGHT = _env_float("PERTURB_LINF_COMPONENT_WEIGHT", 0.7)
RMSE_COMPONENT_WEIGHT = _env_float("PERTURB_RMSE_COMPONENT_WEIGHT", 0.3)
MAX_CHALLENGE_ATTEMPTS = _env_int("PERTURB_MAX_CHALLENGE_ATTEMPTS", 12)
MINER_EXPLORATION_RATIO = _env_float("PERTURB_MINER_EXPLORATION_RATIO", 0.20)
WANDB_ENABLED = _env_bool("PERTURB_WANDB_ENABLED", False)
WANDB_PROJECT = os.getenv("PERTURB_WANDB_PROJECT", "perturb-validator")
WANDB_ENTITY = os.getenv("PERTURB_WANDB_ENTITY", "perturb-ai").strip()
WANDB_RUN_NAME = os.getenv("PERTURB_WANDB_RUN_NAME", "").strip()
WANDB_MODE = os.getenv("PERTURB_WANDB_MODE", "online").strip()
WANDB_LOG_CONSOLE = _env_bool("PERTURB_WANDB_LOG_CONSOLE", True)

VALIDATOR_CONFIG = {
    "image_endpoint": IMAGE_ENDPOINT,
    "pexels_api_key": PEXELS_API_KEY,
    "pexels_per_page": PEXELS_PER_PAGE,
    "pexels_page_span": PEXELS_PAGE_SPAN,
    "pexels_image_variant": PEXELS_IMAGE_VARIANT,
    "image_size": IMAGE_SIZE,
    "timeout_seconds": TIMEOUT_SECONDS,
    "query_interval_seconds": QUERY_INTERVAL_SECONDS,
    "k_miners": K_MINERS,
    "history_size": HISTORY_SIZE,
    "min_processed_count": MIN_PROCESSED_COUNT,
    "llm_endpoint_url": LLM_ENDPOINT_URL,
    "llm_endpoint_model": LLM_ENDPOINT_MODEL,
    "llm_endpoint_timeout_seconds": LLM_ENDPOINT_TIMEOUT_SECONDS,
    "min_linf_delta": MIN_LINF_DELTA,
    "max_linf_delta": MAX_LINF_DELTA,
    "min_ssim": MIN_SSIM,
    "min_psnr_db": MIN_PSNR_DB,
    "linf_component_weight": LINF_COMPONENT_WEIGHT,
    "rmse_component_weight": RMSE_COMPONENT_WEIGHT,
    "max_challenge_attempts": MAX_CHALLENGE_ATTEMPTS,
    "miner_exploration_ratio": MINER_EXPLORATION_RATIO,
    "wandb_enabled": WANDB_ENABLED,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_run_name": WANDB_RUN_NAME,
    "wandb_mode": WANDB_MODE,
    "wandb_log_console": WANDB_LOG_CONSOLE,
}

# Validator scoring defaults.
SPEED_WEIGHT = _env_float("PERTURB_SPEED_WEIGHT", 0.35)
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 0.65)
GAMMA_HISTORY_WEIGHT = _env_float("PERTURB_GAMMA_HISTORY_WEIGHT", 0.7)

