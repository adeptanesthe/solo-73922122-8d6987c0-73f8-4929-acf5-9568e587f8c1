"""灵敏度试验：输入解析、叠加判定、推荐与 API 行为。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.audio import detect_impacts
from app.sensitivity import (
    GAIN_LEVELS_DB,
    SensitivityInputError,
    parse_impact_time,
    parse_trial_times,
    run_sensitivity,
)
from conftest import SR, build_wav


def _idle(duration=5.0, noise=0.01, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration * SR)) * noise).astype(np.float32)


def _sample(impact_time=0.02, duration=0.06, amp=0.8) -> np.ndarray:
    rng = np.random.default_rng(11)
    n = int(duration * SR)
    x = np.zeros(n, dtype=np.float32)
    start = int(impact_time * SR)
    burst = int(0.012 * SR)
    decay = np.exp(-np.linspace(0, 5, burst)).astype(np.float32)
    x[start : start + burst] = (
        amp * rng.standard_normal(burst).astype(np.float32) * decay
    )
    return x


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _post(client, idle, sample, impact_time="0.02", trial_times="1.0,2.0,3.0,4.0"):
    return client.post(
        "/api/sensitivity",
        files={
            "idle": ("idle.wav", idle, "audio/wav"),
            "sample": ("sample.wav", sample, "audio/wav"),
        },
        data={"impact_time": impact_time, "trial_times": trial_times},
    )


# --------------------------- 解析 ---------------------------


def test_parse_impact_time():
    assert parse_impact_time("0.02") == pytest.approx(0.02)
    with pytest.raises(SensitivityInputError):
        parse_impact_time("")
    with pytest.raises(SensitivityInputError):
        parse_impact_time("abc")
    with pytest.raises(SensitivityInputError):
        parse_impact_time("nan")


def test_parse_trial_times_formats():
    assert parse_trial_times("[1.0, 2.0, 3.5]") == [1.0, 2.0, 3.5]
    assert parse_trial_times("1.0, 2.0，3.5 4.2") == [1.0, 2.0, 3.5, 4.2]
    assert parse_trial_times("1.0;2.0;3.5") == [1.0, 2.0, 3.5]
    with pytest.raises(SensitivityInputError):
        parse_trial_times("[1.0, 2.0")
    with pytest.raises(SensitivityInputError):
        parse_trial_times("1.0, foo, 3.0")


# ------------------------ 纯计算判定 ------------------------


def test_gain_levels_cover_minus24_to_zero_step_3():
    assert GAIN_LEVELS_DB == (-24, -21, -18, -15, -12, -9, -6, -3, 0)


def test_all_levels_pass_on_clean_idle_and_recommend_lowest():
    idle = _idle(noise=0.005)
    sample = _sample()
    res = run_sensitivity(idle, sample, 0.02, [1.0, 2.0, 3.0, 4.0])
    assert len(res["levels"]) == len(GAIN_LEVELS_DB)
    assert res["recommendation"]["gain_db"] == -24
    for lv in res["levels"]:
        assert lv["status"] in {"pass", "unmeasurable"}
        if lv["status"] == "pass":
            assert lv["full_mix"]["ok"] and lv["leave_one_out"]["ok"]
            assert lv["failure_reasons"] == []


def test_clipping_levels_are_unmeasurable_and_not_recommended():
    idle = _idle(noise=0.005)
    sample = _sample(amp=0.95)
    res = run_sensitivity(idle, sample, 0.02, [1.0, 2.0, 3.0])
    clipped = [lv for lv in res["levels"] if lv["status"] == "unmeasurable"]
    assert clipped, "高增益档应因超出 PCM 量程不可测"
    for lv in clipped:
        assert lv["full_mix"] is None and lv["leave_one_out"] is None
        assert any("PCM 量程" in r for r in lv["failure_reasons"])
    assert res["recommendation"]["gain_db"] not in {
        lv["gain_db"] for lv in clipped
    }


def test_idle_noise_spike_causes_failure_outside_windows():
    """空转录音自身在试验窗口外的噪声峰 -> 窗外有峰，该档不通过。"""
    idle = _idle(noise=0.0005)
    # 在试验时刻之外放一个很强的噪声峰
    spike = int(2.5 * SR)
    idle[spike : spike + 240] += 0.9
    res = run_sensitivity(_idle_quiet_with_spike(idle), _sample(), 0.02,
                          [1.0, 2.0, 3.5, 4.0])
    failing = [lv for lv in res["levels"] if lv["status"] == "fail"]
    assert failing
    assert any("窗口外检出峰" in r for r in failing[0]["failure_reasons"])


def _idle_quiet_with_spike(idle: np.ndarray) -> np.ndarray:
    return idle


def test_idle_spike_inside_window_fails_leave_one_out():
    """空转噪声在试验时刻窗口内已有峰 -> 撤除样音后仍有峰，不通过。"""
    idle = _idle(noise=0.0005)
    t = 2.0
    start = int(t * SR) - 120  # 窗口内（±15ms）的噪声峰
    idle[start : start + 160] += 0.8
    res = run_sensitivity(idle, _sample(), 0.02, [1.0, 2.0, 3.5, 4.0])
    fail_lvs = [lv for lv in res["levels"] if lv["status"] == "fail"]
    assert fail_lvs
    reasons = " ".join(
        r for lv in fail_lvs for r in lv["failure_reasons"]
    )
    assert "单独撤掉" in reasons or "2 个峰" in reasons


def test_no_level_passes_gives_no_recommendation():
    idle = _idle(noise=0.0005)
    # 每个试验时刻窗口内都放一个与样音无关的强峰
    for t in (1.0, 2.0, 3.0, 4.0):
        start = int(t * SR) - 100
        idle[start : start + 150] += 0.85
    res = run_sensitivity(idle, _sample(), 0.02, [1.0, 2.0, 3.0, 4.0])
    assert res["recommendation"] is None
    assert res["summary"]["pass_count"] == 0
    assert "没有" in res["recommendation_reason"]


def test_sample_too_long_rejected():
    with pytest.raises(SensitivityInputError, match="超过上限"):
        run_sensitivity(_idle(), _sample(duration=0.12), 0.02, [1.0, 2.0, 3.0])


def test_sample_impact_time_out_of_range_rejected():
    with pytest.raises(SensitivityInputError, match="超出样音范围"):
        run_sensitivity(_idle(), _sample(duration=0.05), 0.2, [1.0, 2.0, 3.0])


def test_sample_with_two_impacts_rejected():
    sample = _sample(duration=0.080, amp=0.6)
    src = int(0.02 * SR)
    start = int(0.065 * SR)  # 与 0.02s 的冲击间隔 45ms，不会被回声规则合并
    burst = int(0.012 * SR)
    sample[start : start + burst] += sample[src : src + burst]
    assert len(detect_impacts(sample).times) == 2
    with pytest.raises(SensitivityInputError, match="单次冲击"):
        run_sensitivity(_idle(), sample, 0.02, [1.0, 2.0, 3.0])


def test_silent_sample_rejected():
    with pytest.raises(SensitivityInputError, match="静音"):
        run_sensitivity(_idle(), np.zeros(int(0.05 * SR), dtype=np.float32),
                        0.02, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "trials",
    [
        [1.0, 2.0],                       # 少于 3 个
        [1.0, 2.0, 3.0, 4.0, 4.2, 4.5, 4.8],  # 多于 6 个
        [1.0, 1.1, 2.0, 3.0],             # 间隔 < 150ms
    ],
)
def test_bad_trial_time_counts_and_gaps(trials):
    with pytest.raises(SensitivityInputError):
        run_sensitivity(_idle(), _sample(), 0.02, trials)


def test_trial_time_past_recording_end_rejected():
    with pytest.raises(SensitivityInputError, match="完整落在"):
        run_sensitivity(_idle(5.0), _sample(), 0.02, [1.0, 2.0, 4.99])


def test_trial_time_before_sample_start_rejected():
    sample = _sample(impact_time=0.04, duration=0.07)
    with pytest.raises(SensitivityInputError, match="完整落在"):
        run_sensitivity(_idle(5.0), sample, 0.04, [0.02, 1.0, 2.0])


def test_impact_alignment_uses_filled_time():
    """冲击对齐按填写的样音时钟，偏移量的差异不应造成窗口内多峰。"""
    sample = _sample(impact_time=0.04, duration=0.07)
    res = run_sensitivity(_idle(5.0, noise=0.005), sample, 0.04,
                          [1.0, 2.0, 3.0])
    assert res["recommendation"] is not None
    lv = res["levels"][0]
    assert lv["full_mix"]["ok"]


def test_offset_between_filled_and_detected_time_warns():
    sample = _sample(impact_time=0.02)  # 检测器读到的峰约 0.025s
    res = run_sensitivity(_idle(), sample, 0.033, [1.0, 2.0, 3.0])
    assert any("相差" in w for w in res["warnings"])


# ----------------------------- API -----------------------------


def test_api_success_recommends_and_is_retrievable(client):
    resp = _post(client, _wav(_idle(noise=0.005)), _wav(_sample()))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommendation"]["gain_db"] == -24
    assert len(body["levels"]) == 9
    assert body["idle_url"].endswith(".wav")
    assert client.get(body["idle_url"]).status_code == 200
    last = client.get("/api/sensitivity/last").json()
    assert last["id"] == body["id"]


def test_api_bad_input_retracts_previous_recommendation(client):
    ok = _post(client, _wav(_idle(noise=0.005)), _wav(_sample()))
    assert ok.status_code == 200
    assert ok.json()["recommendation"]["gain_db"] == -24

    bad = _post(client, _wav(_idle()), _wav(_sample()), trial_times="1.0,1.1,2.0")
    assert bad.status_code == 400
    assert "至少相隔" in bad.json()["detail"]

    last = client.get("/api/sensitivity/last").json()
    assert last["status"] == "input_error"
    assert last["recommendation"] is None
    assert "已撤下旧推荐" in last["recommendation_reason"]


def test_api_wrong_wav_format_retracts(client):
    resp = _post(client, build_wav([1.0], sr=8000, duration=2.0), _wav(_sample()))
    assert resp.status_code == 400
    assert "采样率" in resp.json()["detail"]
    assert client.get("/api/sensitivity/last").json()["recommendation"] is None


def test_api_missing_form_field_retracts(client):
    ok = _post(client, _wav(_idle(noise=0.005)), _wav(_sample()))
    assert ok.status_code == 200
    assert ok.json()["recommendation"] is not None

    # 缺少必填的试验时刻：按输入有误处理，并撤下旧推荐
    resp = client.post(
        "/api/sensitivity",
        files={
            "idle": ("idle.wav", _wav(_idle()), "audio/wav"),
            "sample": ("sample.wav", _wav(_sample()), "audio/wav"),
        },
        data={"impact_time": "0.02"},
    )
    assert resp.status_code == 400
    assert "试验时刻" in resp.json()["detail"]
    last = client.get("/api/sensitivity/last").json()
    assert last["recommendation"] is None
    assert last["status"] == "input_error"


def test_api_no_previous_record(client):
    assert client.get("/api/sensitivity/last").status_code == 404


def test_api_missing_file_retracts(client):
    ok = _post(client, _wav(_idle(noise=0.005)), _wav(_sample()))
    assert ok.status_code == 200
    assert ok.json()["recommendation"] is not None

    resp = client.post(
        "/api/sensitivity",
        files={"sample": ("sample.wav", _wav(_sample()), "audio/wav")},
        data={"impact_time": "0.02", "trial_times": "1.0,2.0,3.0"},
    )
    assert resp.status_code == 400
    assert "缺少空转录音" in resp.json()["detail"]
    last = client.get("/api/sensitivity/last").json()
    assert last["recommendation"] is None
    assert last["status"] == "input_error"
    assert "已撤下旧推荐" in last["recommendation_reason"]


def test_sensitivity_does_not_touch_inspect_result(client):
    from conftest import post_inspect

    ref = build_wav([0.5, 1.0, 1.5, 2.0], duration=2.6, noise_floor=0.002)
    good = post_inspect(client, ref, build_wav([0.5, 1.0, 1.5, 2.0], duration=2.6))
    assert good.status_code == 200
    good_id = good.json()["id"]
    _post(client, _wav(_idle()), _wav(_sample()))
    assert client.get("/api/result/last").json()["id"] == good_id
