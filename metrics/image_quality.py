"""Image quality and perturbation-norm metrics for DataCook."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _nchw(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:  # N,H,W grayscale
        return images[:, None, :, :]
    if images.ndim != 4:
        raise ValueError("Images must have shape (N,H,W), (N,C,H,W), or (N,H,W,C).")
    if images.shape[1] in (1, 3):
        return images
    if images.shape[-1] in (1, 3):
        return np.transpose(images, (0, 3, 1, 2))
    raise ValueError("Could not infer image channel dimension; only grayscale and RGB are supported.")


def _ncdhw(volumes: np.ndarray) -> np.ndarray:
    if volumes.ndim != 5:
        raise ValueError("Volumes must have shape (N,C,D,H,W) or (N,D,H,W,C).")
    if volumes.shape[1] in (1, 3):
        return volumes
    if volumes.shape[-1] in (1, 3):
        return np.transpose(volumes, (0, 4, 1, 2, 3))
    raise ValueError(
        "Could not infer volume channel dimension; only one- and three-channel "
        "volumes are supported."
    )


def _canonical_samples(values: np.ndarray) -> tuple[np.ndarray, bool]:
    if values.ndim == 5:
        return _ncdhw(values), True
    return _nchw(values), False


def _summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.all(np.isposinf(array)):
        return float("inf"), 0.0
    if np.all(np.isneginf(array)):
        return float("-inf"), 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _lpips_values(
    raw: np.ndarray,
    cooked: np.ndarray,
    device: str | None,
    batch_size: int = 64,
) -> list[float]:
    """Compute LPIPS in bounded batches, resizing tiny MedMNIST images.

    AlexNet's pooling stack cannot process the native 28x28 MedMNIST
    resolution.  Resizing is applied only for LPIPS; SSIM, PSNR and norm
    metrics continue to use the original pixels.
    """
    if batch_size <= 0:
        raise ValueError("LPIPS batch_size must be positive.")
    try:
        import lpips  # type: ignore
        import torch
        import torch.nn.functional as F
    except ImportError:
        warnings.warn(
            "LPIPS is not installed; lpips_mean/lpips_std will be NaN. Install it with `pip install lpips`.",
            RuntimeWarning,
        )
        return [float("nan")] * raw.shape[0]

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = lpips.LPIPS(net="alex").to(selected_device).eval()
    except OSError as exc:
        raise RuntimeError(f"Failed to load LPIPS model: {exc}") from exc

    raw_tensor = torch.from_numpy(raw)
    cooked_tensor = torch.from_numpy(cooked)
    distances = []
    with torch.no_grad():
        for start in range(0, len(raw_tensor), batch_size):
            raw_batch = raw_tensor[start:start + batch_size].to(selected_device)
            cooked_batch = cooked_tensor[start:start + batch_size].to(selected_device)
            if raw_batch.shape[1] == 1:
                raw_batch = raw_batch.repeat(1, 3, 1, 1)
                cooked_batch = cooked_batch.repeat(1, 3, 1, 1)
            if min(raw_batch.shape[-2:]) < 64:
                raw_batch = F.interpolate(
                    raw_batch, size=(64, 64), mode="bilinear", align_corners=False
                )
                cooked_batch = F.interpolate(
                    cooked_batch, size=(64, 64), mode="bilinear", align_corners=False
                )
            # Input arrays use [0,1]; LPIPS expects [-1,1].
            batch_distances = model(raw_batch * 2 - 1, cooked_batch * 2 - 1)
            distances.append(batch_distances.reshape(-1).detach().cpu())
    return torch.cat(distances).numpy().astype(float).tolist()


def compute_image_quality_metrics(
    raw_images: Any,
    cooked_images: Any,
    *,
    data_range: float | None = None,
    compute_lpips: bool = True,
    device: str | None = None,
) -> dict[str, float]:
    """Return SSIM/PSNR/LPIPS and perturbation-norm summaries.

    Both inputs must already use the same scale. If ``data_range`` is omitted,
    [0,1] and [0,255] inputs are recognized; other ranges use their joint span.
    Images may be NCHW/NHWC and volumes may be NCDHW/NDHWC. For volumes, SSIM
    is computed over all three spatial dimensions and LPIPS must be disabled.

    ``psnr_mean`` is the dataset-level PSNR computed from the aggregate MSE.
    This avoids an undefined infinite arithmetic mean when only a subset of
    quantized images is pixel-identical. ``psnr_std`` is the sample standard
    deviation over finite per-image PSNR values. If every image is identical,
    PSNR remains correctly reported as infinity with zero standard deviation.
    """
    raw, is_volume = _canonical_samples(_numpy(raw_images))
    cooked, cooked_is_volume = _canonical_samples(_numpy(cooked_images))
    if is_volume != cooked_is_volume:
        raise ValueError("raw_images and cooked_images must both be images or both be volumes.")
    if raw.shape != cooked.shape:
        raise ValueError(f"raw_images and cooked_images have different shapes: {raw.shape} vs {cooked.shape}.")
    if raw.shape[0] == 0:
        raise ValueError("At least one image is required.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(cooked)):
        raise ValueError("Images contain NaN or infinite values.")
    joint_min = float(min(raw.min(), cooked.min()))
    joint_max = float(max(raw.max(), cooked.max()))
    if data_range is None:
        if joint_min >= 0 and joint_max <= 1.0 + 1e-6:
            data_range = 1.0
        elif joint_min >= 0 and joint_max <= 255.0 + 1e-6:
            data_range = 255.0
        else:
            data_range = joint_max - joint_min
    if data_range <= 0:
        data_range = 1.0

    ssim_values, psnr_values, mse_values, l2_values, linf_values = [], [], [], [], []
    for raw_sample, cooked_sample in zip(raw, cooked):
        if is_volume:
            raw_spatial = np.transpose(raw_sample, (1, 2, 3, 0))
            cooked_spatial = np.transpose(cooked_sample, (1, 2, 3, 0))
            spatial_shape = raw_spatial.shape[:3]
        else:
            raw_spatial = np.transpose(raw_sample, (1, 2, 0))
            cooked_spatial = np.transpose(cooked_sample, (1, 2, 0))
            spatial_shape = raw_spatial.shape[:2]
        win_size = min(7, *spatial_shape)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            sample_type = "volumes" if is_volume else "images"
            raise ValueError(f"SSIM requires {sample_type} at least 3 pixels per spatial dimension.")
        channel_axis = -1 if raw_spatial.shape[-1] > 1 else None
        if channel_axis is None:
            raw_spatial = raw_spatial[..., 0]
            cooked_spatial = cooked_spatial[..., 0]
        ssim_values.append(float(structural_similarity(
            raw_spatial,
            cooked_spatial,
            data_range=data_range,
            channel_axis=channel_axis,
            win_size=win_size,
        )))
        difference = (cooked_sample - raw_sample).reshape(-1)
        mse = float(np.mean(np.square(difference.astype(np.float64))))
        mse_values.append(mse)
        if mse > 0:
            psnr_values.append(float(10.0 * np.log10((data_range ** 2) / mse)))
        l2_values.append(float(np.linalg.norm(difference, ord=2)))
        linf_values.append(float(np.max(np.abs(difference))))

    if compute_lpips and is_volume:
        raise ValueError(
            "LPIPS is a 2D image metric and is not supported for 3D volumes; "
            "set compute_lpips=False."
        )
    if compute_lpips:
        normalized_raw = np.clip((raw - joint_min) / data_range, 0.0, 1.0)
        normalized_cooked = np.clip((cooked - joint_min) / data_range, 0.0, 1.0)
        lpips_values = _lpips_values(normalized_raw, normalized_cooked, device)
    else:
        lpips_values = [float("nan")] * raw.shape[0]

    result: dict[str, float] = {}
    for name, values in (
        ("ssim", ssim_values),
        ("lpips", lpips_values),
        ("l2_norm", l2_values),
        ("linf_norm", linf_values),
    ):
        if np.all(np.isnan(values)):
            mean, std = float("nan"), float("nan")
        else:
            mean, std = _summary(values)
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
    aggregate_mse = float(np.mean(mse_values))
    if aggregate_mse == 0:
        result["psnr_mean"] = float("inf")
        result["psnr_std"] = 0.0
    else:
        result["psnr_mean"] = float(10.0 * np.log10((data_range ** 2) / aggregate_mse))
        result["psnr_std"] = _summary(psnr_values)[1]
    result["fid"] = float("nan")
    return result
