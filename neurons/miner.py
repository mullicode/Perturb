import argparse
import logging as pylogging
import os
import time
import typing
import math
import asyncio

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import (
    _preprocess_for_efficientnet_v2_l,
    load_efficientnet_v2_l,
    resolve_target_index,
)
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)

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


def _finalize_perturbed_image(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_label: int,
    epsilon: float,
) -> typing.Tuple[torch.Tensor, _AdvQuality, bool, str, int]:
    """Run preflight checks and final encode/decode verification on a candidate image."""
    decoded_adv, quality = _quality_on_png(model, clean, adv, true_label, true_label)
    preflight = _preflight_flip_only(quality, true_label, epsilon)

    encoded_b64 = encode_image_b64(decoded_adv)
    decoded_final = decode_image_b64(encoded_b64).to(clean.device)
    pred_final = _pred_index(model, decoded_final)

    if pred_final == true_label:
        logger.warning(
            f"[SUBMIT] FLIP LOST after final encode/decode! "
            f"pred={pred_final} true={true_label} — submitting anyway"
        )
    else:
        logger.info(
            f"[SUBMIT] encode/decode flip confirmed pred={pred_final} true={true_label}"
        )

    return decoded_adv, quality, preflight.ok, preflight.reason, pred_final


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
 
        # Top-20 candidate target classes by logit (excluding true class)
        top_classes = lg.detach().argsort(descending=True)
        top_classes = [
            c.item() for c in top_classes
            if c.item() != true_idx
        ][:20]
 
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
            return None
 
        targets.sort(key=lambda c: c["estimated_rmse"])
        best_target = targets[0]
        target_idx = best_target["target"]
        k_est = best_target["k_estimated"]
        logger.info(
            f"[TARGET] best={target_idx} "
            f"est_rmse={best_target['estimated_rmse']:.2e} "
            f"k_est={k_est}"
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
 
        def get_batch(k_cur, cur_gap):
            """Adaptive batch size from progress and current confidence."""
            progress = k_cur / max(k_est, 1)
            gap_ratio = cur_gap / max(gap_initial, 1e-8)
            if progress < 0.03:          # first 3%: exact, batch=1
                return 1
            if progress > 0.80:          # last 20%: near boundary, tiny batch
                return max(1, min(5, int(gap_ratio * 10)))
            base = (200 if gap_ratio > 0.7 else
                    100 if gap_ratio > 0.4 else
                     30 if gap_ratio > 0.2 else 10)
            return max(5, int(base * (1.0 - progress)))
 
        while len(selected) < int(k_est * 1.5):
            if time.perf_counter() > deadline - 1.5:
                break
 
            # Gradient refresh at current adversarial state
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
 
            batch_size = get_batch(len(selected), cur_gap)
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
                if margin < -1.0:   # deep enough — stop early
                    break
 
        if best_result is None:
            logger.warning(
                f"[ATTACK] No flip found after {len(selected)} pixels"
            )
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
        rmse    = ((k_final / N) ** 0.5) / 255.0
 
        logger.info(
            f"[FINAL] k={k_final} rmse={rmse:.2e} "
            f"norm={eps:.4f} margin={best_result['margin']:.3f}"
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
        if synapse.norm_type != "Linf":
            logger.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        task_id = getattr(synapse, "task_id", "unknown")
        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        true_idx = resolve_target_index(synapse.true_label)
        if true_idx is None:
            logger.warning(
                f"Skipping task={task_id}: unresolved true_label={getattr(synapse, 'true_label', None)}"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = 1.0/255.0
        min_delta = float(getattr(synapse, "min_delta", 1.0/255.0))
        budget = 13.0
        t0 = time.perf_counter()
        deadline = t0 + budget
        c, h, w = clean.shape
        logger.info(
            f"[FORWARD] task_eps={epsilon:.4f} res={c}x{h}x{w} "
            f"val_linf=[{_VAL_MIN_LINF:.4f},{min(epsilon, _VAL_MAX_LINF):.4f}]"
        )
        try:
            result = self._attack(
                clean, true_idx, epsilon, min_delta, deadline
            )
        except Exception as e:
            logger.warning(f"Attack failed with exception: {e}")
            result = None
        if result is None:
            logger.warning("No flip found — returning clean image")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse
        loop = asyncio.get_event_loop()
        adv = clean
        (
            decoded_adv,
            quality,
            preflight_ok,
            preflight_reason,
            pred_final,
        ) = await loop.run_in_executor(
            None,
            lambda: _finalize_perturbed_image(
                model=self.model,
                clean=clean,
                adv=result["image"],
                true_label=true_idx,
                epsilon=epsilon,
            ),
        )
        t_enc0 = time.perf_counter()
        synapse.perturbed_image_b64 = encode_image_b64(decoded_adv)
        encode_ms = (time.perf_counter() - t_enc0) * 1000.0
        logger.info(
            f"Finished task={task_id} "
            f"true={true_idx} pred={quality.pred} "
            f"flip={quality.flipped} l_inf={quality.norm:.5f} rmse={quality.rmse:.2e} "
            f"ssim={quality.ssim:.4f} psnr={quality.psnr_db:.2f} "
            f"preflight={preflight_ok} reason={preflight_reason} "
            f"final_pred={pred_final} "
            f"encode_ms={encode_ms:.1f} "
            f"total_ms={(time.perf_counter() - t0) * 1000:.1f}"
        )

        del adv, clean, decoded_adv
        if os.getenv("MINER_CUDA_EMPTY_CACHE", "0").strip() == "1":
            torch.cuda.empty_cache()

        return synapse

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

