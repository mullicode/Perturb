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
        # ── Phase 2: adaptive greedy ───────────────────────────────────────
        def fit_decay_model(gap_history):
            if len(gap_history) < 3:
                return None, None, float('inf'), 0

            points = [(p, g) for p, g, _ in gap_history if g > 1e-6]
            if len(points) < 3:
                return None, None, float('inf'), 0


            # ── KEY FIX: work in relative coordinates ─────────────────
            # Instead of fitting gap(k) = gap_0 * exp(-lam * k) in absolute k,
            # normalize x to [0, 1] range to prevent overflow in math.exp(a).
            # This makes the intercept a = ln(gap at x=0) which is bounded.
            k_min = points[0][0]
            k_max = max(points[-1][0], k_min + 1)
            k_range = k_max - k_min

            n = len(points)
            weights = []
            non_monotone_count = 0
            for i in range(n):
                if i == 0:
                    weights.append(1.0)
                    continue
                _, g_prev = points[i-1]
                _, g_cur  = points[i]
                if g_cur >= g_prev:
                    non_monotone_count += 1
                    weights.append(0.1)
                else:
                    recency = 1.0 + 2.0 * (i / max(n - 1, 1))
                    weights.append(recency)

            # Normalize x to [0,1] — prevents exp overflow entirely
            xs = [(p[0] - k_min) / k_range for p in points]
            ys = [math.log(max(p[1], 1e-10)) for p in points]
            ws = weights[:n]

            sw   = sum(ws)
            swx  = sum(w * x for w, x in zip(ws, xs))
            swy  = sum(w * y for w, y in zip(ws, ys))
            swxx = sum(w * x * x for w, x in zip(ws, xs))
            swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

            denom = sw * swxx - swx * swx
            if abs(denom) < 1e-12:
                return None, None, float('inf'), 0

            b = (sw * swxy - swx * swy) / denom   # slope in normalized space
            a = (swy - b * swx) / sw               # intercept = ln(gap at k_min)

            # lam in normalized space → convert back to per-pixel lam
            # In normalized space: gap(x) = exp(a) * exp(b*x), x = (k-k_min)/k_range
            # In pixel space:      gap(k) = exp(a) * exp(b*(k-k_min)/k_range)
            #                             = exp(a) * exp((b/k_range)*(k-k_min))
            # So lam_per_pixel = -b / k_range
            lam = -b / k_range   # always in per-pixel units

            # gap_0 at k=k_min — bounded because a = ln(gap at k_min) ≈ ln(gap_history values)
            # gap values are in [1e-6, ~10], so a is in [-14, +2.3] — always safe
            gap_at_k_min = math.exp(max(min(a, 50.0), -50.0))   # hard clamp as safety net

            residuals = [abs(ys[i] - (a + b * xs[i])) for i in range(n)]
            residual = sum(residuals) / n
            non_monotone_fraction = non_monotone_count / max(n - 1, 1)
            adjusted_residual = residual * (1.0 + 2.0 * non_monotone_fraction)

            # Return lam and gap_at_k_min along with k_min for use in prediction
            return lam, gap_at_k_min, adjusted_residual, k_min
        
        def fit_linear_model(gap_history):
            """Weighted linear rate fallback. Returns float or None."""
            if len(gap_history) < 2:
                return None
            wsum, wtot = 0.0, 0.0
            for i in range(1, len(gap_history)):
                p0, g0, _ = gap_history[i - 1]
                p1, g1, _ = gap_history[i]
                dp = max(p1 - p0, 1)
                dg = g0 - g1
                if dg > 1e-9:
                    w = 3.0 if i >= len(gap_history) - 1 else (
                        2.0 if i >= len(gap_history) - 2 else 1.0
                    )
                    wsum += w * (dg / dp)
                    wtot += w
            return (wsum / wtot) if wtot > 0 else None

        def predict_pixels_to_flip(cur_gap, lam, gap_at_k_min, k_min, k_cur):
            if lam <= 1e-10:
                return float('inf')
            # Threshold must be truly 0 (flip boundary).
            # Use 0.001 — well below where check_flip triggers,
            # so pixels_needed stays non-zero until we're truly at boundary.
            # Scaling with gap_initial caused underestimation for large-gap images.
            FLIP_THRESHOLD = 0.001
            if cur_gap <= FLIP_THRESHOLD:
                return 1.0
            delta = math.log(cur_gap / FLIP_THRESHOLD) / lam
            return max(delta, 1.0)
        
        def get_batch(k_cur, cur_gap, last_batch_time):
            nonlocal is_first_cycle
            if is_first_cycle:
                is_first_cycle = False
                return CALIBRATION_BATCH

            time_remaining = deadline - 0.5 - time.perf_counter()
            if time_remaining <= 0:
                return 1

            gap_fraction = cur_gap / max(gap_initial, 1e-8)

            # ── Model fitting ─────────────────────────────────────────
            fit_result = fit_decay_model(gap_history)
            # fit_decay_model now returns 4 values
            if fit_result[0] is not None:
                lam, gap_at_k_min, exp_residual, k_min_fit = fit_result
            else:
                lam, gap_at_k_min, exp_residual, k_min_fit = None, None, float('inf'), 0

            linear_rate = fit_linear_model(gap_history)

            use_exponential = (
                lam is not None and
                lam > 1e-8 and
                exp_residual < 0.15 and
                len(gap_history) >= 3
            )

            if use_exponential:
                pixels_needed = predict_pixels_to_flip(
                    cur_gap, lam, gap_at_k_min, k_min_fit, k_cur
                )
                model_name = "exp"
                natural_batch = max(1, int(0.0513 / lam))
            elif linear_rate is not None and linear_rate > 1e-9:
                pixels_needed = cur_gap / linear_rate
                model_name = "linear"
                natural_batch = None
            else:
                pixels_needed = max(k_est - k_cur, 1.0)
                model_name = "fallback"
                natural_batch = None

            # ── Time budget ───────────────────────────────────────────
            # last_batch_time = cost of one full cycle (gradient + selection)
            # This is the direct measurement, not an estimate
            cost_per_cycle = max(last_batch_time, 0.02)
            cycles_remaining = time_remaining / cost_per_cycle

            # NEW
            # pace_batch = pixels this cycle must deliver to finish on time
            # = pixels_needed spread evenly across remaining cycles
            pace_batch = max(1, int(math.ceil(pixels_needed / max(cycles_remaining, 1.0))))

            # time_ratio = pace_batch / natural_batch
            # > 1: behind pace (need bigger batches than natural)
            # < 1: ahead of pace (natural batch is fine)
            if natural_batch is not None:
                time_ratio = pace_batch / max(natural_batch, 1)
            else:
                time_ratio = 1.0

            if use_exponential and lam > 1e-10:
                natural_batch = max(1, int(0.0513 / lam))
                if time_ratio <= 1.0:
                    batch = natural_batch
                else:
                    # Scale up proportionally to how far behind we are
                    batch = max(natural_batch, int(math.ceil(natural_batch * time_ratio)))
            else:
                # Linear or fallback: just use pace directly
                batch = pace_batch

            # Hard floor: pace_batch is the minimum required to finish in time.
            # natural_batch or time_ratio logic may produce a smaller value when
            # last_batch_time was stale or lam is small — this guarantees we
            # never fall below the pace required to reach the flip before deadline.
            batch = max(batch, pace_batch)

            # ── Near-boundary precision: rate-derived, never arbitrary ─
            # When gap is small, compute maximum batch such that
            # the expected gap after this batch stays > 0 with margin.
            # If batch is too large, we skip over the boundary without detecting flip.
            if gap_fraction < 0.05 and use_exponential and lam > 1e-10:
                # Only cap when truly near boundary (gap < 5% of initial)
                # Allow up to 80% gap reduction per batch
                max_overshoot_batch = max(1, int(1.609 / lam))
                batch = min(batch, max_overshoot_batch)
            elif gap_fraction < 0.05 and linear_rate is not None and linear_rate > 1e-9:
                max_drop_batch = max(1, int(cur_gap * 0.80 / linear_rate))
                batch = min(batch, max_drop_batch)

            if len(gap_history) >= 3:
                recent = gap_history[-3:]
                non_mono = sum(
                    1 for i in range(1, len(recent))
                    if recent[i][1] >= recent[i-1][1]
                )
                if non_mono >= 2:
                    batch = max(1, batch // 2)

            batch = max(batch, 1)


            lam_str = f"{lam:.2e}" if lam else "None"
            residual_str = (
                f"{exp_residual:.3f}"
                if exp_residual != float("inf")
                else "inf"
            )
            logger.info(
                f"[GET_BATCH] batch={batch} gap={cur_gap:.4f} frac={gap_fraction:.3f} "
                f"model={model_name} lam={lam_str} residual={residual_str} "
                f"pixels_needed={int(pixels_needed)} pace={pace_batch} "
                f"natural={natural_batch if natural_batch is not None else 'N/A'} "
                f"time_ratio={time_ratio:.3f} time_left={time_remaining:.2f}"
            )
            return batch

        def check_flip(adv_float):
            """Verify flip on uint8-roundtripped image."""
            snapped = (adv_float * 255.0).round().clamp(0, 255) / 255.0
            with torch.no_grad():
                lg_s = logits_for_images(
                    self.model, snapped.unsqueeze(0)
                ).squeeze(0)
                pred = int(lg_s.argmax().item())
                margin = (lg_s[true_idx] - lg_s[pred]).item()
            return pred != true_idx, margin, snapped

        current_adv = clean.clone()
        selected = []                                        # pixel indices
        selected_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        selected_signs = torch.zeros(N, dtype=torch.long, device=self.device)
        best_result = None
        gap_initial = gap
        cur_gap = gap
        g_t = best_target["gradient"]
        g_norm1 = g_t.abs().sum().item()
        lambda_prior = (eps * g_norm1) / (gap_initial * N)

        CALIBRATION_BATCH = max(5, int(0.10 / max(lambda_prior, 1e-8)))
        CALIBRATION_BATCH = min(CALIBRATION_BATCH, 200)
        is_first_cycle = True

        gap_history = [(0, gap_initial, time.perf_counter())]
        last_batch_time = 0.20
        while True:
            if time.perf_counter() > deadline - 0.5:
                break
            cycle_start = time.perf_counter()
            x_in = current_adv.detach().requires_grad_(True)
            lg_in = logits_for_images(self.model, x_in.unsqueeze(0)).squeeze(0)

            # Compute gap from the SAME forward pass — no extra forward needed
            with torch.no_grad():
                cur_gap = (lg_in[true_idx] - lg_in.topk(2).values[1]).item()

            # Two backward passes sharing the same retained graph
            lg_in[true_idx].backward(retain_graph=True)
            g_true_cur = x_in.grad.detach().reshape(-1).clone()

            x_in.grad.zero_()
            lg_in[target_idx].backward()
            g_tgt_cur = x_in.grad.detach().reshape(-1).clone()

            g_cur = g_true_cur - g_tgt_cur
            
            gap_history.append((len(selected), max(cur_gap, 1e-6), time.perf_counter()))
            if len(gap_history) > 12:
                gap_history.pop(0)

            # If we've applied 1.5x k_est pixels with no flip,
            # gap is stuck (diminishing returns on current target).
            # Break to avoid wasting remaining budget on exhausted pixels.
            if len(selected) > k_est * 1.5 and best_result is None and cur_gap > 0.05:
                logger.warning(
                    f"[ATTACK] Exceeded 1.5x k_est={k_est} at k={len(selected)} "
                    f"gap={cur_gap:.4f} — stopping, greedy exhausted"
                )
                break

            if cur_gap < 2.0:
                _flipped, _margin, _snapped = check_flip(current_adv)
                if _flipped:
                    _, flip_q = _quality_on_png(
                        self.model, clean, _snapped, true_idx, true_idx
                    )
                    preflight = _preflight_flip_only(flip_q, true_idx, challenge_epsilon)
                    if preflight.ok and (best_result is None or flip_q.rmse < best_result.get("rmse_val", float("inf"))):
                        pred_now = int(flip_q.pred)
                        flip_score = _estimate_validator_score(
                            flip_q,
                            true_idx,
                            challenge_epsilon,
                            (time.perf_counter() - attack_t0) * 1000.0,
                            timeout_seconds,
                        )
                        best_result = {
                            "image":    _snapped.clone(),
                            "k":        len(selected),
                            "margin":   _margin,
                            "selected": selected.copy(),
                            "pred":     pred_now,
                            "rmse_val": flip_q.rmse,
                        }
                        log_attack(
                            f"[FLIP] k={len(selected)} margin={_margin:.3f}",
                            prediction=pred_now,
                            rmse=flip_q.rmse,
                            norm=flip_q.norm,
                            estimated_score=flip_score,
                        )
                        if flip_q.rmse < 0.3 * _VAL_MAX_LINF:
                            break

            batch_size = get_batch(len(selected), cur_gap, last_batch_time)
            # Scores for this step
            flat_cur = current_adv.reshape(-1)
            sc = g_cur.abs()
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
            n_pick = min(batch_size, int(valid_cur.sum().item()))
            if n_pick == 0:
                break

            # ── Near-boundary: direct Δgap measurement per candidate ──────
            # When gap is tiny and batch is small, the gradient ranking can
            # misorder the last few pixels. Directly measuring actual gap delta
            # per forward pass finds the minimum-k flip path.
            # Cost: n_measure × 1 forward pass. Only runs when it matters.
            if cur_gap < 0.05 and n_pick <= 5:
                try:
                    _n_direct = min(n_pick * 6, 30)
                    _direct_cands = torch.topk(sc, min(_n_direct, int((sc > 0).sum().item()))).indices
                    if len(_direct_cands) >= n_pick:
                        _base_u8 = (current_adv * 255.0).round().clamp(0, 255).long()
                        with torch.no_grad():
                            _lg_base = logits_for_images(self.model, current_adv.unsqueeze(0)).squeeze(0)
                            _gap_base = (_lg_base[true_idx] - _lg_base[target_idx]).item()
                        _delta_gaps = []
                        for _ci in _direct_cands.tolist():
                            _test_u8 = _base_u8.clone()
                            _flat_test = _test_u8.reshape(-1)
                            # Use the boundary gradient sign for this pixel
                            _sign = -int(g_cur[_ci].sign().item())
                            if _sign == 0:
                                _sign = 1
                            _flat_test[_ci] = (_flat_test[_ci] + _sign).clamp(0, 255)
                            _test_img = _flat_test.reshape(C, H, W).float() / 255.0
                            with torch.no_grad():
                                _lg_t = logits_for_images(self.model, _test_img.unsqueeze(0)).squeeze(0)
                                _dgap = (_lg_t[true_idx] - _lg_t[target_idx]).item() - _gap_base
                            _delta_gaps.append((_ci, _dgap))
                        # Most negative Δgap = most gap reduction = best pixel
                        _delta_gaps.sort(key=lambda z: z[1])
                        top_idx = torch.tensor(
                            [z[0] for z in _delta_gaps[:n_pick]],
                            dtype=torch.long, device=self.device
                        )
                    else:
                        top_idx = torch.topk(sc, n_pick).indices
                except Exception:
                    top_idx = torch.topk(sc, n_pick).indices
            else:
                top_idx = torch.topk(sc, n_pick).indices

            # Vectorized — replaces entire for loop
            s_vec = signs_cur[top_idx].long()          # signs for selected pixels
            adv_u8 = (current_adv * 255.0).round().clamp(0, 255).long()
            flat_u8 = adv_u8.reshape(-1)
            flat_u8[top_idx] = (flat_u8[top_idx] + s_vec).clamp(0, 255)
            current_adv = flat_u8.reshape(C, H, W).float() / 255.0

            # Update selected list and mask
            selected.extend(top_idx.tolist())
            selected_mask[top_idx] = True
            selected_signs[top_idx] = s_vec
            last_batch_time = 0.7 * last_batch_time + 0.3 * max(time.perf_counter() - cycle_start, 0.02)
            # Flip check on uint8 roundtrip
            if best_result is None:
                _flipped2, _margin2, _snapped2 = check_flip(current_adv)
                if _flipped2:
                    _, flip_q2 = _quality_on_png(
                        self.model, clean, _snapped2, true_idx, true_idx
                    )
                    preflight2 = _preflight_flip_only(flip_q2, true_idx, challenge_epsilon)
                    if preflight2.ok:
                        best_result = {
                            "image":    _snapped2.clone(),
                            "k":        len(selected),
                            "margin":   _margin2,
                            "selected": selected.copy(),
                            "pred":     int(flip_q2.pred),
                            "rmse_val": flip_q2.rmse,
                        }

        if best_result is None:
            logger.warning(
                f"[ATTACK] No flip found after {len(selected)} pixels"
            )
            log_attack(f"[ATTACK] No flip found after {len(selected)} pixels")
            return None
        # ── Phase 4: backward elimination ─────────────────────────────────
        elim_budget = min(1.2, deadline - time.perf_counter() - 0.3)
        elim_deadline = time.perf_counter() + elim_budget
        # Store actual sign used when pixel was selected
        sel = best_result["selected"].copy()
        curr = best_result["image"].clone()

        def still_flips(adv_float):
            try:
                decoded = _encode_decode_roundtrip(adv_float)
            except Exception:
                decoded = (adv_float * 255.0).round().clamp(0, 255) / 255.0
            with torch.no_grad():
                lg2 = logits_for_images(
                    self.model, decoded.unsqueeze(0)
                ).squeeze(0)
                pred2 = int(lg2.argmax().item())
                marg = (lg2[true_idx] - lg2[pred2]).item()
            if lg2.argmax().item() == true_idx:
                return False
            q = _measure_adv_quality(self.model, clean, decoded, true_idx, true_idx)
            return _preflight_flip_only(q, true_idx, challenge_epsilon).ok

        def apply_sel(sel_list):
            """Rebuild adversarial image from selection list."""
            base = (clean * 255.0).round().clamp(0, 255).long()
            flat_b  = base.reshape(-1)
            sel_t = torch.tensor(sel_list, dtype=torch.long, device=self.device)
            s = selected_signs[sel_t]          # actual signs used at selection time
            flat_b[sel_t] = (flat_b[sel_t] + s).clamp(0, 255)
            return flat_b.reshape(C, H, W).float() / 255.0
 
        improved = True
        logger.info(f"[ELIM] k_initial={len(sel)} elim_budget={elim_budget}")
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

