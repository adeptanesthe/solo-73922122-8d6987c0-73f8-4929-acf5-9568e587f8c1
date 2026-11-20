"""空转噪声下冲击检出灵敏度试验。

检修员换过麦克风位置后，需要知道「空转噪声下多弱的冲击还能被检出」：

1. 空转录音保持原音量；
2. 单冲样音里的冲击对准每个试验时刻叠加到空转录音上，样音增益
   从 -24 dB 到 0 dB 每 3 dB 试一档；
3. 混音峰值超出 16-bit PCM 可表示量程的档位记为不可测（unmeasurable）；
4. 用现有冲击检测器（``app.audio.detect_impacts``）判定每一档：
   - 完整混音：每个试验时刻前后 15 ms 内恰好检出 1 个峰，窗外无峰；
   - 逐处撤除：单独撤掉某处样音后，该处窗口内无峰；
   两条同时满足才算该档通过；
5. 推荐通过的最低增益档；无档通过时不给推荐。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

import numpy as np

from .audio import SAMPLE_RATE, detect_impacts

# 样音时长上限（单冲样音，不超过 80ms）
SAMPLE_MAX_DURATION_SEC = 0.080
# 峰判定窗口：试验时刻前后 15ms
WINDOW_SEC = 0.015
# 增益档位：-24 dB ~ 0 dB，每 3 dB 一档
GAIN_LEVELS_DB: tuple[int, ...] = tuple(range(-24, 1, 3))
# 试验时刻数量与最小间隔
MIN_TRIAL_COUNT = 3
MAX_TRIAL_COUNT = 6
MIN_TRIAL_GAP_SEC = 0.150
# 填写的样音冲击时刻与样音内检出冲击的允许偏差
IMPACT_TIME_TOL_SEC = 0.015
# 填写偏差超过该值时给出提示（会直接决定对齐基准）
IMPACT_TIME_WARN_SEC = 0.005
# 16-bit PCM 可表示量程：[-1.0, 32767/32768]
PCM_MIN = -1.0
PCM_MAX = 32767 / 32768
EPS = 1e-9


class SensitivityInputError(ValueError):
    """灵敏度试验的输入不符合约定。"""


def parse_impact_time(text: str | None) -> float:
    """解析网页/API 传入的「样音中冲击的时刻」（秒）。"""
    if text is None or not str(text).strip():
        raise SensitivityInputError("请填写样音中冲击的时刻（秒）")
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise SensitivityInputError(
            f"样音冲击时刻必须是数字（秒），当前：{text!r}"
        ) from exc
    if not math.isfinite(value):
        raise SensitivityInputError("样音冲击时刻必须是有限数值")
    return value


def parse_trial_times(text: str | None) -> list[float]:
    """解析试验时刻列表。

    支持 JSON 数组（``[1.0, 2.0, 3.0]``）或逗号/空白分隔
    （``1.0, 2.0, 3.0``）。
    """
    if text is None or not str(text).strip():
        raise SensitivityInputError("请填写 3～6 个试验时刻（秒）")
    raw = str(text).strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SensitivityInputError(
                "试验时刻不是合法的 JSON 数组"
            ) from exc
        if not isinstance(parsed, list):
            raise SensitivityInputError("试验时刻必须是数组")
        items: Iterable[Any] = parsed
    else:
        items = re.split(r"[,，;；\s]+", raw)

    values: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float, str)):
            raise SensitivityInputError(f"试验时刻包含非法值：{item!r}")
        try:
            value = float(str(item).strip())
        except (TypeError, ValueError) as exc:
            raise SensitivityInputError(
                f"试验时刻必须是数字（秒），当前：{item!r}"
            ) from exc
        if not math.isfinite(value):
            raise SensitivityInputError("试验时刻必须是有限数值")
        values.append(value)
    return values


def _validate_trial_times(
    trial_times: list[float],
    *,
    idle_duration: float,
    sample_duration: float,
    sample_impact_time: float,
    n_idle: int,
    n_sample: int,
) -> list[float]:
    if not (MIN_TRIAL_COUNT <= len(trial_times) <= MAX_TRIAL_COUNT):
        raise SensitivityInputError(
            f"试验时刻须为 {MIN_TRIAL_COUNT}～{MAX_TRIAL_COUNT} 个，"
            f"当前 {len(trial_times)} 个"
        )

    ordered = sorted(trial_times)
    for earlier, later in zip(ordered, ordered[1:]):
        if later - earlier < MIN_TRIAL_GAP_SEC - EPS:
            raise SensitivityInputError(
                f"相邻试验时刻至少相隔 {int(MIN_TRIAL_GAP_SEC * 1000)} ms；"
                f"当前 {earlier:g}s 与 {later:g}s 间隔仅 "
                f"{(later - earlier) * 1000:.0f} ms"
            )

    # 秒级可放置范围：冲击对齐后，样音片段须完整落在空转录音内
    earliest = sample_impact_time
    latest = idle_duration - (sample_duration - sample_impact_time)
    for t in ordered:
        if t < -EPS or t > idle_duration + EPS:
            raise SensitivityInputError(
                f"试验时刻 {t:g}s 超出空转录音范围 0～{idle_duration:.3f}s"
            )
        start = int(round((t - sample_impact_time) * SAMPLE_RATE))
        if start < 0 or start + n_sample > n_idle:
            raise SensitivityInputError(
                f"试验时刻 {t:g}s 处样音片段无法完整落在空转录音内；"
                f"该样音可放置的试验时刻范围约为 "
                f"{max(0.0, earliest):.3f}～{max(0.0, latest):.3f}s"
            )
    return ordered


def _mix(
    idle_samples: np.ndarray,
    sample_samples: np.ndarray,
    starts: list[int],
    gain_linear: float,
) -> np.ndarray:
    mixed = idle_samples.astype(np.float32, copy=True)
    n_sample = len(sample_samples)
    for start in starts:
        mixed[start : start + n_sample] += sample_samples * gain_linear
    return mixed


def _window_hits(peaks: list[float], trial_time: float) -> list[float]:
    return [p for p in peaks if abs(p - trial_time) <= WINDOW_SEC + EPS]


def run_sensitivity(
    idle_samples: np.ndarray,
    sample_samples: np.ndarray,
    sample_impact_time: float,
    trial_times: list[float],
) -> dict[str, Any]:
    """执行全部增益档的叠加与判定，返回可直接序列化的结果。

    该函数为纯计算：不读写文件、不保存状态。
    """
    n_idle = int(len(idle_samples))
    n_sample = int(len(sample_samples))
    if n_idle == 0:
        raise SensitivityInputError("空转录音为空，没有任何采样点")
    if n_sample == 0:
        raise SensitivityInputError("样音为空，没有任何采样点")

    idle_duration = n_idle / SAMPLE_RATE
    sample_duration = n_sample / SAMPLE_RATE

    if sample_duration > SAMPLE_MAX_DURATION_SEC + EPS:
        raise SensitivityInputError(
            f"样音时长 {sample_duration * 1000:.1f} ms 超过上限 "
            f"{SAMPLE_MAX_DURATION_SEC * 1000:.0f} ms"
        )
    if not math.isfinite(sample_impact_time):
        raise SensitivityInputError("样音冲击时刻必须是有限数值")
    if sample_impact_time < -EPS or sample_impact_time > sample_duration + EPS:
        raise SensitivityInputError(
            f"样音冲击时刻 {sample_impact_time:g}s 超出样音范围 "
            f"0～{sample_duration:.3f}s"
        )
    if not np.any(np.abs(sample_samples) > 0):
        raise SensitivityInputError("样音为静音，没有可叠加的冲击能量")

    warnings: list[str] = []
    sample_det = detect_impacts(sample_samples)
    if len(sample_det.times) > 1:
        raise SensitivityInputError(
            f"样音应为单次冲击，当前检测到 {len(sample_det.times)} 次冲击"
        )
    if len(sample_det.times) == 1:
        delta = abs(sample_det.times[0] - sample_impact_time)
        if delta > IMPACT_TIME_TOL_SEC + EPS:
            raise SensitivityInputError(
                f"样音中检测到的冲击时刻为 {sample_det.times[0]:.3f}s，"
                f"与填写的 {sample_impact_time:.3f}s 相差 {delta * 1000:.0f} ms"
                f"（允许 {IMPACT_TIME_TOL_SEC * 1000:.0f} ms），请核对"
            )
        if delta > IMPACT_TIME_WARN_SEC + EPS:
            warnings.append(
                f"填写的冲击时刻与样音中检出时刻相差 {delta * 1000:.0f} ms，"
                "本次按填写时刻对齐"
            )
    else:
        warnings.append(
            "样音中未检出明确的单次冲击，已按填写的冲击时刻对齐叠加"
        )

    trials = _validate_trial_times(
        trial_times,
        idle_duration=idle_duration,
        sample_duration=sample_duration,
        sample_impact_time=sample_impact_time,
        n_idle=n_idle,
        n_sample=n_sample,
    )
    # 片段起点按「试验时刻 − 样音冲击时刻」计算，冲击落在试验时刻上
    starts = [int(round((t - sample_impact_time) * SAMPLE_RATE)) for t in trials]

    levels: list[dict[str, Any]] = []
    for gain_db in GAIN_LEVELS_DB:
        gain_linear = 10.0 ** (gain_db / 20.0)
        full_mix = _mix(idle_samples, sample_samples, starts, gain_linear)
        mixed_peak = float(np.max(np.abs(full_mix)))

        # 超出 PCM 量程：不能实际回放/判定，记为不可测
        if (
            float(np.max(full_mix)) > PCM_MAX + 1e-6
            or float(np.min(full_mix)) < PCM_MIN - 1e-6
        ):
            levels.append(
                {
                    "gain_db": gain_db,
                    "gain_linear": round(gain_linear, 6),
                    "status": "unmeasurable",
                    "mixed_peak": round(mixed_peak, 6),
                    "failure_reasons": [
                        f"混音峰值 {mixed_peak:.4f} 超出 16-bit PCM 量程 "
                        "[-1.0, 1.0)，该档无法实际回放，记为不可测"
                    ],
                    "full_mix": None,
                    "leave_one_out": None,
                }
            )
            continue

        full_peaks = detect_impacts(full_mix).times

        windows: list[dict[str, Any]] = []
        full_reasons: list[str] = []
        for trial_time in trials:
            hits = _window_hits(full_peaks, trial_time)
            ok = len(hits) == 1
            windows.append(
                {
                    "trial_time": round(trial_time, 4),
                    "detected": [round(p, 4) for p in hits],
                    "peak_count": len(hits),
                    "ok": ok,
                }
            )
            if len(hits) == 0:
                full_reasons.append(
                    f"试验时刻 {trial_time:.3f}s 的 ±{int(WINDOW_SEC * 1000)} ms "
                    "窗口内未检出峰"
                )
            elif len(hits) > 1:
                full_reasons.append(
                    f"试验时刻 {trial_time:.3f}s 的 ±{int(WINDOW_SEC * 1000)} ms "
                    f"窗口内检出 {len(hits)} 个峰（要求恰好 1 个）"
                )

        outside = [
            p
            for p in full_peaks
            if not any(abs(p - t) <= WINDOW_SEC + EPS for t in trials)
        ]
        if outside:
            shown = "、".join(f"{p:.3f}" for p in outside[:8])
            more = f" 等 {len(outside)} 个" if len(outside) > 8 else ""
            full_reasons.append(f"所有试验窗口外检出峰：{shown}{more}")
        full_ok = not full_reasons

        removals: list[dict[str, Any]] = []
        loo_reasons: list[str] = []
        # 撤除一处 = 该处恢复为原音量空转录音；空转录音本身来自 16-bit PCM，
        # 必然在量程内，因此撤除后的混音也一定可回放，无需再判量程。
        for k, trial_time in enumerate(trials):
            remaining = [s for j, s in enumerate(starts) if j != k]
            removed_mix = _mix(
                idle_samples, sample_samples, remaining, gain_linear
            )
            removed_peaks = detect_impacts(removed_mix).times
            hits = _window_hits(removed_peaks, trial_time)
            ok = not hits
            removals.append(
                {
                    "trial_time": round(trial_time, 4),
                    "detected": [round(p, 4) for p in hits],
                    "peak_count": len(hits),
                    "ok": ok,
                }
            )
            if hits:
                loo_reasons.append(
                    f"单独撤掉 {trial_time:.3f}s 处的样音后，该窗口内仍检出 "
                    f"{len(hits)} 个峰，说明该峰并非来自样音"
                )
        loo_ok = not loo_reasons

        passed = full_ok and loo_ok
        status = "pass" if passed else "fail"
        reasons = full_reasons + loo_reasons
        levels.append(
            {
                "gain_db": gain_db,
                "gain_linear": round(gain_linear, 6),
                "status": status,
                "mixed_peak": round(mixed_peak, 6),
                "failure_reasons": reasons,
                "full_mix": {
                    "ok": full_ok,
                    "windows": windows,
                    "outside_peaks": [round(p, 4) for p in outside],
                },
                "leave_one_out": {
                    "ok": loo_ok,
                    "removals": removals,
                },
            }
        )

    pass_levels = [lv["gain_db"] for lv in levels if lv["status"] == "pass"]
    fail_count = sum(1 for lv in levels if lv["status"] == "fail")
    unmeasurable_count = sum(
        1 for lv in levels if lv["status"] == "unmeasurable"
    )
    if pass_levels:
        recommended = min(pass_levels)
        recommendation_reason = (
            f"通过的最低增益档为 {recommended:+d} dB（增益越低，"
            "能检出的冲击越弱）"
        )
    else:
        recommended = None
        if unmeasurable_count == len(levels):
            recommendation_reason = (
                "所有档位的混音都超出 PCM 量程，均不可测，无法给出推荐"
            )
        else:
            recommendation_reason = (
                f"没有任何增益档通过（{fail_count} 档未通过，"
                f"{unmeasurable_count} 档超出量程不可测），已撤下原推荐"
            )

    return {
        "sample_duration": round(sample_duration, 4),
        "idle_duration": round(idle_duration, 4),
        "sample_impact_time": round(sample_impact_time, 4),
        "trial_times": [round(t, 4) for t in trials],
        "window_ms": int(WINDOW_SEC * 1000),
        "gain_levels_db": list(GAIN_LEVELS_DB),
        "levels": levels,
        "recommendation": (
            None
            if recommended is None
            else {"gain_db": recommended, "reason": recommendation_reason}
        ),
        "recommendation_reason": recommendation_reason,
        "summary": {
            "pass_count": len(pass_levels),
            "fail_count": fail_count,
            "unmeasurable_count": unmeasurable_count,
        },
        "warnings": warnings,
    }
