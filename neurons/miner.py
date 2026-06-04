import argparse
import datetime
import logging as pylogging
import os
import threading
import time
import typing
import math
import asyncio

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from openpyxl import Workbook, load_workbook

from perturbnet.model import (
    _preprocess_for_efficientnet_v2_l,
    load_efficientnet_v2_l,
    logits_for_images,
    resolve_target_index,
)
from perturbnet.protocol import AttackChallenge
from perturbnet import constants as C

logger = pylogging.getLogger(__name__)

_ATTACK_EXCEL_LOCK = threading.Lock()
_ATTACK_EXCEL_HEADERS = (
    "timestamp",
    "task_id",
    "model_name",
    "prompt",
    "true_label",
    "epsilon",
    "norm_type",
    "min_delta",
    "resolution",
    "prediction",
    "progress",
    "rmse",
    "norm",
    "estimated_score",
)


def _append_attack_excel_row(
    excel_path: str,
    *,
    task_id: str,
    model_name: str,
    prompt: str,
    true_label: str,
    epsilon: float,
    norm_type: str,
    min_delta: float,
    resolution: str,
    progress: str = "",
    prediction: typing.Optional[int] = None,
    rmse: typing.Optional[float] = None,
    norm: typing.Optional[float] = None,
    estimated_score: typing.Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(excel_path)) or ".", exist_ok=True)
    row = [
        datetime.datetime.now().isoformat(timespec="seconds"),
        task_id,
        model_name,
        prompt,
        true_label,
        epsilon,
        norm_type,
        min_delta,
        resolution,
        "" if prediction is None else prediction,
        progress,
        "" if rmse is None else rmse,
        "" if norm is None else norm,
        "" if estimated_score is None else estimated_score,
    ]
    with _ATTACK_EXCEL_LOCK:
        if os.path.isfile(excel_path):
            wb = load_workbook(excel_path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "attack_log"
            ws.append(list(_ATTACK_EXCEL_HEADERS))
        ws.append(row)
        wb.save(excel_path)


class _AttackExcelRecorder:
    """Accumulates attack progress and writes one Excel row per task."""

    def __init__(
        self,
        excel_path: str,
        *,
        task_id: str,
        model_name: str,
        prompt: str,
        true_label: str,
        epsilon: float,
        norm_type: str,
        min_delta: float,
        resolution: str,
    ) -> None:
        self._excel_path = excel_path
        self._task_id = task_id
        self._model_name = model_name
        self._prompt = prompt
        self._true_label = true_label
        self._epsilon = epsilon
        self._norm_type = norm_type
        self._min_delta = min_delta
        self._resolution = resolution
        self._progress: list[str] = []
        self._prediction: typing.Optional[int] = None
        self._rmse: typing.Optional[float] = None
        self._norm: typing.Optional[float] = None
        self._estimated_score: typing.Optional[float] = None

    def log(
        self,
        progress: str,
        prediction: typing.Optional[int] = None,
        rmse: typing.Optional[float] = None,
        norm: typing.Optional[float] = None,
        estimated_score: typing.Optional[float] = None,
    ) -> None:
        self._progress.append(progress)
        if prediction is not None:
            self._prediction = prediction
        if rmse is not None:
            self._rmse = rmse
        if norm is not None:
            self._norm = norm
        if estimated_score is not None:
            self._estimated_score = estimated_score

    def set_resolution(self, resolution: str) -> None:
        self._resolution = resolution

    def flush(self) -> None:
        if not self._progress:
            return
        _append_attack_excel_row(
            self._excel_path,
            task_id=self._task_id,
            model_name=self._model_name,
            prompt=self._prompt,
            true_label=self._true_label,
            epsilon=self._epsilon,
            norm_type=self._norm_type,
            min_delta=self._min_delta,
            resolution=self._resolution,
            progress=" | ".join(self._progress),
            prediction=self._prediction,
            rmse=self._rmse,
            norm=self._norm,
            estimated_score=self._estimated_score,
        )


# ── Validator acceptance window ─────────────────────────────────────────────
# These MUST mirror neurons/validator.py::verify_and_score + the canonical
# perturbnet/constants.py defaults. They are VALIDITY gates (what the validator
# will actually accept). The validator's L-inf window is [min_linf, min(epsilon,
# max_linf)] = [0.003, 0.03] for the sampled epsilon range [0.06, 0.20], and it
# never rejects on RMSE (RMSE only affects the score).
_VAL_MIN_LINF      = float(os.getenv("MINER_VAL_MIN_LINF",    "0.003"))
_VAL_MAX_LINF      = float(os.getenv("MINER_VAL_MAX_LINF",    "0.00393"))
_VAL_MIN_SSIM      = float(os.getenv("MINER_VAL_MIN_SSIM",    "0.98"))
_VAL_MIN_PSNR_DB   = float(os.getenv("MINER_VAL_MIN_PSNR_DB", "38.0"))

# ── Flip-robustness margin ──────────────────────────────────────────────────
# A flip is only accepted when the true label sits at least this many logits
# BELOW the winning class. A pure argmax flip (margin ≈ 0) is a boundary flip
# that does not survive the small numerical differences between this miner's
# inference path and the validator's (TF32 conv, cuDNN algo choice, device,
# bicubic-resize numerics) — the validator then re-classifies it as the
# original label (reason=label_match_with_original, score=0).
_FLIP_MARGIN_EPS   = float(os.getenv("MINER_FLIP_MARGIN_EPS", "0.4"))
# Match the validator's numerics by disabling TF32 reductions on the miner so
# a locally-confirmed flip reflects what the validator will actually compute.
_DISABLE_TF32      = os.getenv("MINER_DISABLE_TF32", "1").strip() not in ("0", "false", "no")
if _DISABLE_TF32:
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Model forward helpers
# ═══════════════════════════════════════════════════════════════════════════

def _model_logits_batch(model: torch.nn.Module, x_in: torch.Tensor) -> torch.Tensor:
    x_prep = _preprocess_for_efficientnet_v2_l(x_in)
    return model(x_prep)


def _pred_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    with torch.inference_mode():
        logits = _model_logits_batch(model, image_chw.unsqueeze(0).float())
        return int(logits.argmax(dim=1)[0].item())


# ═══════════════════════════════════════════════════════════════════════════
#  Bittensor plumbing
# ═══════════════════════════════════════════════════════════════════════════

def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config) -> typing.Any:
    axon_cfg = getattr(config, "axon", None)
    port     = int(
        getattr(axon_cfg, "port", None)
        or getattr(config, "axon_port", None)
        or os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))
    )
    ip = str(
        getattr(axon_cfg, "ip", None) or os.getenv("MINER_IP", os.getenv("AXON_IP", "0.0.0.0"))
    ).strip() or "0.0.0.0"
    external_ip = str(
        getattr(axon_cfg, "external_ip", None) or os.getenv("MINER_EXTERNAL_IP", "")
    ).strip()
    external_port_raw = (
        getattr(axon_cfg, "external_port", None) or os.getenv("MINER_EXTERNAL_PORT", "")
    )
    external_port = int(str(external_port_raw).strip()) if str(external_port_raw).strip() else port
    if not external_ip:
        raise RuntimeError(
            "MINER_EXTERNAL_IP is not set. "
            "Set it to the public IP address that validators can reach this miner on."
        )
    max_workers = int(getattr(axon_cfg, "max_workers", None) or os.getenv("AXON_MAX_WORKERS", "10"))
    axon_cls    = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("bittensor.Axon class not found.")
    logger.info(
        f"[MINER] Creating axon ip={ip} port={port} "
        f"external_ip={external_ip} external_port={external_port} max_workers={max_workers}"
    )
    return axon_cls(
        wallet=wallet, ip=ip, port=port,
        external_ip=external_ip, external_port=external_port,
        max_workers=max_workers,
    )


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)

# ═══════════════════════════════════════════════════════════════════════════
#  Quality / preflight / PNG submit
# ═══════════════════════════════════════════════════════════════════════════

class _AdvQuality(typing.NamedTuple):
    norm:       float
    rmse:       float
    ssim:       float
    psnr_db:    float
    pred:       int
    target_hit: bool
    flipped:    bool
    # logits[true] - max_{j != true} logits[j].  < 0 ⇒ argmax flipped;
    # <= -_FLIP_MARGIN_EPS ⇒ robust (transferable) flip.
    margin:     float = 0.0


class _PreflightResult(typing.NamedTuple):
    ok:      bool
    reason:  str
    quality: _AdvQuality


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3 or x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x, y    = x_clean.unsqueeze(0), x_adv.unsqueeze(0)
    c1, c2  = 0.01 ** 2, 0.03 ** 2
    mu_x    = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y    = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x  = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator   = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _measure_adv_quality(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_label: int,
    target_label: int,
) -> _AdvQuality:
    diff = adv - clean
    norm = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
    tl = int(true_label)
    with torch.inference_mode():
        logits = _model_logits_batch(model, adv.unsqueeze(0).float()).squeeze(0).float()
    pred = int(logits.argmax().item())
    # margin = logits[true] - best competing logit.  A robust flip needs the
    # true label to sit at least _FLIP_MARGIN_EPS below the winner, otherwise
    # the boundary flip will not transfer to the validator's inference path.
    competitor = logits.clone()
    competitor[tl] = float("-inf")
    margin = float(logits[tl].item() - competitor.max().item())
    robust_flip = (pred != tl) and (margin <= -_FLIP_MARGIN_EPS)
    return _AdvQuality(
        norm=norm, rmse=rmse,
        ssim=_compute_ssim(clean, adv), psnr_db=_compute_psnr_db(clean, adv),
        pred=pred, target_hit=pred == int(target_label), flipped=robust_flip,
        margin=margin,
    )


def _encode_decode_roundtrip(adv: torch.Tensor) -> torch.Tensor:
    """Single PNG encode then decode — matches exactly what validator does."""
    return decode_image_b64(encode_image_b64(adv)).to(adv.device)


def _quality_on_png(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_label: int,
    target_label: int,
) -> typing.Tuple[torch.Tensor, _AdvQuality]:
    """Measure quality against the original clean image via one PNG roundtrip."""
    decoded = _encode_decode_roundtrip(adv)
    q = _measure_adv_quality(model, clean, decoded, true_label, target_label)
    return decoded, q


def _preflight_flip_only(
    quality: _AdvQuality,
    true_label: int,
    epsilon: float,
) -> _PreflightResult:
    tl = int(true_label)
    # Mirror neurons/validator.py::verify_and_score acceptance gates exactly.
    # norm/rmse/ssim/psnr are already measured the same way the validator
    # measures them (both clean and adv decoded from PNG), so these checks
    # predict the validator outcome faithfully. The validator's L-inf ceiling
    # is min(epsilon, max_linf_delta) — NOT the miner's 1/255 optimization
    # target — and it has no RMSE rejection.
    eff_max = min(float(epsilon), _VAL_MAX_LINF)
    if quality.pred == tl:
        return _PreflightResult(False, "label_match_with_original", quality)
    if quality.norm < _VAL_MIN_LINF:
        return _PreflightResult(False, "below_min_delta", quality)
    if quality.norm > eff_max:
        return _PreflightResult(False, "above_max_delta", quality)
    if quality.ssim < _VAL_MIN_SSIM:
        return _PreflightResult(False, "below_min_ssim", quality)
    if _VAL_MIN_PSNR_DB > 0.0 and quality.psnr_db < _VAL_MIN_PSNR_DB:
        return _PreflightResult(False, "below_min_psnr_db", quality)
    return _PreflightResult(True, "ok_flip", quality)


def _estimate_validator_score(
    quality: _AdvQuality,
    true_label: int,
    epsilon: float,
    response_time_ms: float,
    timeout_seconds: float,
) -> float:
    """Mirror neurons/validator.py::verify_and_score scoring formula."""
    preflight = _preflight_flip_only(quality, true_label, epsilon)
    if not preflight.ok:
        return 0.0

    effective_max_delta = min(float(epsilon), _VAL_MAX_LINF)
    denom = max(1e-12, effective_max_delta - _VAL_MIN_LINF)
    linf_ratio = min(max((quality.norm - _VAL_MIN_LINF) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2

    rmse_ratio = min(max(quality.rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2

    total_weight = max(1e-12, C.LINF_COMPONENT_WEIGHT + C.RMSE_COMPONENT_WEIGHT)
    perturbation_score = (
        (C.LINF_COMPONENT_WEIGHT * linf_score) + (C.RMSE_COMPONENT_WEIGHT * rmse_score)
    ) / total_weight

    time_ratio = response_time_ms / max(1e-12, timeout_seconds * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)
    return float(C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score)

# ═══════════════════════════════════════════════════════════════════════════
#  Miner
# ═══════════════════════════════════════════════════════════════════════════

class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_v2_l(self.device)

        self._in_flight: dict[str, asyncio.Future] = {}
        self._in_flight_lock = asyncio.Lock()

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self._attack_excel_path = os.path.join(
            getattr(getattr(config, "logging", None), "logging_dir", "./logs"),
            "attack_log.xlsx",
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    def _attack(
        self,
        clean: torch.Tensor,        # float C×H×W [0,1]
        true_idx: int,
        eps: float,                  # 1/255
        min_delta: float,
        deadline: float,
        log_attack: typing.Callable[..., None],
        challenge_epsilon: float,
        timeout_seconds: float,
        attack_t0: float,
    ) -> typing.Optional[dict]:
        """
        Full pipeline. Returns dict with image/k/rmse/norm/margin/pred
        or None if no flip found.
        """
        C, H, W = clean.shape
        N = C * H * W

        # ── Phase 1: Model probe ──────────────────────────────
        x = clean.detach().requires_grad_(True)
        # PREPROCESS handles resize+normalize internally
        lg = logits_for_images(self.model, x.unsqueeze(0)).squeeze(0)

        # Verify model predicts true_idx
        pred_clean = int(lg.argmax().item())
        if pred_clean != true_idx:
            logger.warning(
                f"Model predicts {pred_clean} != true_idx {true_idx} "
                f"on clean image — attacking anyway"
            )

        true_logit = lg[true_idx]
        gap = (true_logit - lg.topk(2).values[1]).item()
        logger.info(f"[PROBE] true={true_idx} gap={gap:.3f} N={N}")
        log_attack(f"[PROBE] true={true_idx} gap={gap:.3f} N={N}")
        # Top-20 candidate target classes by logit (excluding true class)
        top_classes = lg.detach().argsort(descending=True)
        top_classes = [
            c.item() for c in top_classes
            if c.item() != true_idx
        ][:20]
        logger.info(f"[TOP20CLASS] top_classes={top_classes}")

        # Compute true-label gradient once, reuse across all target rankings
        if x.grad is not None:
            x.grad.zero_()
        true_logit.backward(retain_graph=True)
        grad_true = x.grad.detach().reshape(-1).clone()
 
        # Rank targets by estimated_rmse = margin / (||g_t||2 × sqrt(N))
        targets = []
        for i, t in enumerate(top_classes):
            if x.grad is not None:
                x.grad.zero_()
            retain = (i < len(top_classes) - 1)
            lg[t].backward(retain_graph=retain)
            grad_t_raw = x.grad.detach().reshape(-1).clone()
 
            # Boundary gradient for this target
            g_t = grad_true - grad_t_raw
            margin_t = (true_logit - lg[t]).item()
            g_norm2 = g_t.norm(2).item()
            g_norm1 = g_t.abs().sum().item()
 
            if g_norm2 < 1e-10:
                continue
 
            # Feasibility: can 1/255 perturbations reach this boundary?
            if (eps * g_norm1) < margin_t * 0.1:
                continue
 
            estimated_rmse = margin_t / (g_norm2 * (N ** 0.5))
 
            # Estimated pixel count using top-1% gradient mean
            n_top = max(100, int(N * 0.01))
            top_mean = g_t.abs().topk(n_top).values.mean().item()
            k_est = int(margin_t / max(eps * top_mean, 1e-10))
 
            targets.append({
                "target":         t,
                "margin":         margin_t,
                "estimated_rmse": estimated_rmse,
                "k_estimated":    k_est,
                "gradient":       g_t,
            })
 
        if not targets:
            logger.warning("[ATTACK] No feasible targets found")
            log_attack("[ATTACK] No feasible targets found")
            return None
 
        targets.sort(key=lambda c: c["estimated_rmse"])
        best_target = targets[0]
        target_idx = best_target["target"]
        k_est = max(int(best_target["k_estimated"] * 2.0), 500)
        logger.info(
            f"[TARGET] best={target_idx} "
            f"est_rmse={best_target['estimated_rmse']:.2e} "
            f"k_est={k_est}"
        )
        log_attack(
            f"[TARGET] best={target_idx} "
            f"est_rmse={best_target['estimated_rmse']:.2e} k_est={k_est}"
        )
        # ── Phase 2: pixel ranking ─────────────────────────────────────────
        g_tgt = best_target["gradient"]
        signs = -g_tgt.sign()
        signs[signs == 0] = 1.0
 
        # grad × input: SE-aware pixel importance score
        scores = (g_tgt * clean.reshape(-1)).abs()
 
        # valid_dir: only pixels where the gradient-required direction
        # is actually achievable after uint8 clipping
        flat_clean = clean.reshape(-1)
        q_flat = (flat_clean * 255.0).round()
        valid_dir = (
            ((signs > 0) & (q_flat < 255)) |
            ((signs < 0) & (q_flat > 0))
        )
        scores[~valid_dir] = 0.0
 
        # ── Phase 3: adaptive greedy ───────────────────────────────────────
        current_adv = clean.clone()
        selected = []                                        # pixel indices
        selected_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        best_result = None
        gap_initial = gap
 
        def check_flip(adv_float):
            """Verify flip on uint8-roundtripped image."""
            snapped = (adv_float * 255.0).round().clamp(0, 255) / 255.0
            with torch.no_grad():
                lg_s = logits_for_images(
                    self.model, snapped.unsqueeze(0)
                ).squeeze(0)
                pred = int(lg_s.argmax().item())
                margin = (lg_s[true_idx] - lg_s.topk(2).values[1]).item()
            return pred != true_idx, margin, snapped
 
        def get_batch(k_cur, cur_gap, refresh_interval_seconds):
            progress  = k_cur / max(k_est, 1)
            gap_ratio = max(cur_gap, 0.0) / max(gap_initial, 1e-8)

            time_remaining   = deadline - 1.5 - time.perf_counter()
            pixels_remaining = max(k_est - k_cur, 1)

            # Correct iters_remaining: how many gradient refreshes left
            refreshes_left = time_remaining / max(refresh_interval_seconds, 0.05)
            min_batch = int(pixels_remaining / max(refreshes_left, 1))

            # Near-boundary precision cap
            # When gap_ratio < 0.15, model is close to flipping — be precise
            if gap_ratio < 0.05:
                precision_cap = 2
            elif gap_ratio < 0.15:
                precision_cap = 5
            elif gap_ratio < 0.3:
                precision_cap = 15
            else:
                precision_cap = 9999  # no cap

            # Near-completion precision cap
            # When progress > 0.95, slow down
            if progress > 0.95:
                progress_cap = 3
            elif progress > 0.85:
                progress_cap = 10
            else:
                progress_cap = 9999

            effective_cap = min(precision_cap, progress_cap)
            result = max(min_batch, 5)           # floor: always at least 5
            result = min(result, effective_cap)  # ceiling from precision
            result = max(result, min_batch)      # but never starve time budget
            logger.info(f"[GET_BATCH] batch_count={result} current_gap={cur_gap} progress={progress} gap_ratio={gap_ratio}")
            return result

 
        REFRESH_INTERVAL = 50
        refresh_interval_seconds = 0.2  # initial guess
        last_refresh_wall = time.perf_counter()
        g_cur = None
        cur_gap = gap
        last_refresh_count = -REFRESH_INTERVAL
        while len(selected) < int(k_est * 3):
            if time.perf_counter() > deadline - 1.5:
                break
 
            # Gradient refresh at current adversarial state
            if len(selected) - last_refresh_count >= REFRESH_INTERVAL:
                now = time.perf_counter()
                measured = now - last_refresh_wall
                if measured < 5.0 and last_refresh_count >= 0:
                    refresh_interval_seconds = 0.7 * refresh_interval_seconds + 0.3 * measured
                last_refresh_wall = now
                last_refresh_count = len(selected)
                # do gradient refresh ...
                x_in = current_adv.detach().requires_grad_(True)
                lg_in = logits_for_images(
                    self.model, x_in.unsqueeze(0)
                ).squeeze(0)
    
                if x_in.grad is not None:
                    x_in.grad.zero_()
                lg_in[true_idx].backward(retain_graph=True)
                g_true_cur = x_in.grad.detach().reshape(-1).clone()
    
                if x_in.grad is not None:
                    x_in.grad.zero_()
                lg_in[target_idx].backward()
                g_tgt_cur = x_in.grad.detach().reshape(-1).clone()
    
                g_cur = g_true_cur - g_tgt_cur
    
                with torch.no_grad():
                    cur_gap = (
                        lg_in[true_idx] - lg_in.topk(2).values[1]
                    ).item()
                last_refresh_count = len(selected)
                if cur_gap <= 0:
                    flipped, margin, snapped = check_flip(current_adv)
                    if flipped:
                        if best_result is None or len(selected) < best_result["k"]:
                            with torch.no_grad():
                                pred_now = int(
                                    logits_for_images(
                                        self.model, snapped.unsqueeze(0)
                                    ).argmax().item()
                                )
                            best_result = {
                                "image":    snapped.clone(),
                                "k":        len(selected),
                                "margin":   margin,
                                "selected": selected.copy(),
                                "pred":     pred_now,
                            }
                        break

            # Scores for this step
            flat_cur = current_adv.reshape(-1)
            sc = (g_cur * flat_cur).abs()
            sc[selected_mask] = 0.0
 
            # Direction validity at current state
            q_cur = (flat_cur * 255.0).round()
            signs_cur = -g_cur.sign()
            signs_cur[signs_cur == 0] = 1.0
            valid_cur = (
                ((signs_cur > 0) & (q_cur < 255)) |
                ((signs_cur < 0) & (q_cur > 0))
            )
            sc[~valid_cur] = 0.0
 
            if sc.max() <= 0:
                break
            batch_size = get_batch(len(selected), cur_gap, refresh_interval_seconds)
            n_pick = min(batch_size, int(valid_cur.sum().item()))
            if n_pick == 0:
                break
 
            top_idx = torch.topk(sc, n_pick).indices
 
            # Apply perturbation in uint8 integer space
            # Avoids floating point drift that causes roundtrip failures
            adv_u8 = (current_adv * 255.0).round().clamp(0, 255)
            flat_u8 = adv_u8.reshape(-1)
            for idx in top_idx.tolist():
                s = int(signs_cur[idx].item())
                flat_u8[idx] = (flat_u8[idx] + s).clamp(0, 255)
                selected.append(idx)
                selected_mask[idx] = True
            current_adv = flat_u8.reshape(C, H, W) / 255.0
 
            # Flip check on uint8 roundtrip
            flipped, margin, snapped = check_flip(current_adv)
            if flipped and margin < -0.4:
                if best_result is None or len(selected) < best_result["k"]:
                    with torch.no_grad():
                        pred_now = int(
                            logits_for_images(
                                self.model, snapped.unsqueeze(0)
                            ).argmax().item()
                        )
                    best_result = {
                        "image":    snapped.clone(),
                        "k":        len(selected),
                        "margin":   margin,
                        "selected": selected.copy(),
                        "pred":     pred_now,
                    }
                    _, flip_q = _quality_on_png(
                        self.model, clean, snapped, true_idx, true_idx
                    )
                    flip_score = _estimate_validator_score(
                        flip_q,
                        true_idx,
                        challenge_epsilon,
                        (time.perf_counter() - attack_t0) * 1000.0,
                        timeout_seconds,
                    )
                    log_attack(
                        f"[FLIP] k={len(selected)} margin={margin:.3f}",
                        prediction=pred_now,
                        rmse=flip_q.rmse,
                        norm=flip_q.norm,
                        estimated_score=flip_score,
                    )
                if margin < -1.0:   # deep enough — stop early
                    break
        if best_result is None:
            logger.warning(
                f"[ATTACK] No flip found after {len(selected)} pixels"
            )
            log_attack(f"[ATTACK] No flip found after {len(selected)} pixels")
            return None
        # ── Phase 4: backward elimination ─────────────────────────────────
        elim_budget = min(1.2, deadline - time.perf_counter() - 0.3)
        elim_deadline = time.perf_counter() + elim_budget
 
        sel = best_result["selected"].copy()
        curr = best_result["image"].clone()
 
        def still_flips(adv_float):
            sn = (adv_float * 255.0).round().clamp(0, 255) / 255.0
            with torch.no_grad():
                lg2 = logits_for_images(
                    self.model, sn.unsqueeze(0)
                ).squeeze(0)
                marg = (lg2[true_idx] - lg2.topk(2).values[1]).item()
            return lg2.argmax().item() != true_idx and marg < -0.4
 
        def apply_sel(sel_list):
            """Rebuild adversarial image from selection list."""
            base = (clean * 255.0).round().clamp(0, 255)
            flat_b  = base.reshape(-1)
            for px in sel_list:
                s = int(signs[px].item())
                flat_b[px] = (flat_b[px] + s).clamp(0, 255)
            return flat_b.reshape(C, H, W) / 255.0
 
        improved = True
        logger.info(f"[ELIM] k_initial={len(sel)} elim_deadline={elim_deadline}")
        while improved and time.perf_counter() < elim_deadline:
            improved = False
 
            # Rank pixels by weakest gradient contribution — try removing first
            x_e = curr.detach().requires_grad_(True)
            lg_e = logits_for_images(
                self.model, x_e.unsqueeze(0)
            ).squeeze(0)
            lg_e[true_idx].backward()
            g_e = x_e.grad.detach().reshape(-1).abs()
            order = sorted(
                range(len(sel)),
                key=lambda i: g_e[sel[i]].item()
            )
 
            for pos in order:
                if time.perf_counter() > elim_deadline:
                    break
                trial = sel[:pos] + sel[pos + 1:]
                trial_adv = apply_sel(trial)
                if still_flips(trial_adv):
                    sel = trial
                    curr = trial_adv
                    improved = True
                    break   # restart with updated gradient
 
        k_final = len(sel)
        logger.info(f"[ELIM] k_final={k_final}")
        rmse    = ((k_final / N) ** 0.5) / 255.0

        _, final_q = _quality_on_png(self.model, clean, curr, true_idx, true_idx)
        final_score = _estimate_validator_score(
            final_q,
            true_idx,
            challenge_epsilon,
            (time.perf_counter() - attack_t0) * 1000.0,
            timeout_seconds,
        )

        logger.info(
            f"[FINAL] k={k_final} rmse={rmse:.2e} "
            f"norm={eps:.4f} margin={best_result['margin']:.3f} "
            f"est_score={final_score:.4f}"
        )
        log_attack(
            f"[FINAL] k={k_final} margin={best_result['margin']:.3f}",
            prediction=best_result["pred"],
            rmse=final_q.rmse,
            norm=final_q.norm,
            estimated_score=final_score,
        )
        return {
            "image":  curr,
            "k":      k_final,
            "rmse":   rmse,
            "norm":   eps,
            "margin": best_result["margin"],
            "pred":   best_result["pred"],
        }

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        self._log_step_start(
            "miner_forward",
            task_id=getattr(synapse, "task_id", "unknown"),
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )

        task_id = getattr(synapse, "task_id", "unknown")
        model_name = getattr(synapse, "model_name", "")
        prompt = getattr(synapse, "prompt", "")
        true_label = getattr(synapse, "true_label", "")
        synapse_epsilon = float(getattr(synapse, "epsilon", 0.0))
        norm_type = getattr(synapse, "norm_type", "")
        min_delta = float(getattr(synapse, "min_delta", 1.0 / 255.0))

        resolution = "unknown"
        attack_log = _AttackExcelRecorder(
            self._attack_excel_path,
            task_id=task_id,
            model_name=model_name,
            prompt=prompt,
            true_label=true_label,
            epsilon=synapse_epsilon,
            norm_type=norm_type,
            min_delta=min_delta,
            resolution=resolution,
        )

        def log_attack(
            progress: str,
            prediction: typing.Optional[int] = None,
            rmse: typing.Optional[float] = None,
            norm: typing.Optional[float] = None,
            estimated_score: typing.Optional[float] = None,
        ) -> None:
            attack_log.log(
                progress,
                prediction=prediction,
                rmse=rmse,
                norm=norm,
                estimated_score=estimated_score,
            )

        try:
            if synapse.norm_type != "Linf":
                logger.info(f"Skipping task={task_id}: unsupported norm_type={synapse.norm_type}")
                log_attack(f"[SKIP] unsupported norm_type={synapse.norm_type}")
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                return synapse

            clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
            true_idx = resolve_target_index(synapse.true_label)
            c, h, w = clean.shape
            resolution = f"{c}x{h}x{w}"
            attack_log.set_resolution(resolution)
            log_attack("[START] attack forward")

            if true_idx is None:
                logger.warning(
                    f"Skipping task={task_id}: unresolved true_label={getattr(synapse, 'true_label', None)}"
                )
                log_attack(f"[SKIP] unresolved true_label={true_label}")
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                return synapse

            epsilon = 1.0/255.0
            budget = 13.0
            t0 = time.perf_counter()
            deadline = t0 + budget
            timeout_seconds = float(getattr(synapse, "timeout_seconds", C.TIMEOUT_SECONDS))
            logger.info(
                f"[FORWARD] task_eps={epsilon:.4f} res={c}x{h}x{w} "
                f"val_linf=[{_VAL_MIN_LINF:.4f},{min(epsilon, _VAL_MAX_LINF):.4f}]"
            )
            try:
                result = self._attack(
                    clean,
                    true_idx,
                    epsilon,
                    min_delta,
                    deadline,
                    log_attack,
                    challenge_epsilon=synapse_epsilon,
                    timeout_seconds=timeout_seconds,
                    attack_t0=t0,
                )
            except Exception as e:
                logger.warning(f"Attack failed with exception: {e}")
                log_attack(f"[ERROR] attack exception: {e}")
                result = None
            if result is None:
                logger.warning("No flip found — returning clean image")
                log_attack("[DONE] no flip — returning clean image", estimated_score=0.0)
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                return synapse

            synapse.perturbed_image_b64 = encode_image_b64(result["image"])
            _, submit_q = _quality_on_png(
                self.model, clean, result["image"], true_idx, true_idx
            )
            submit_score = _estimate_validator_score(
                submit_q,
                true_idx,
                synapse_epsilon,
                (time.perf_counter() - t0) * 1000.0,
                timeout_seconds,
            )
            logger.info(
                f"Finished task={task_id} "
                f"true={true_idx} pred={result['pred']} "
                f"k={result['k']} rmse={submit_q.rmse:.2e} "
                f"norm={submit_q.norm:.4f} margin={result['margin']:.3f} "
                f"est_score={submit_score:.4f}"
            )
            log_attack(
                f"[DONE] submitted k={result['k']} margin={result['margin']:.3f}",
                prediction=result["pred"],
                rmse=submit_q.rmse,
                norm=submit_q.norm,
                estimated_score=submit_score,
            )
            return synapse
        finally:
            attack_log.flush()

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

