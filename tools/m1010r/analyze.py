"""Hash-verified digital calibration analysis, never candidate or physical A/V."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .timing_analysis import analyze_audio, analyze_frames
else:
    from timing_analysis import analyze_audio, analyze_frames


ROOT = Path(__file__).resolve().parents[2]


def artifact(reference):
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT / ".cache/m1010r"):
        raise ValueError("Raw artifact outside R cache")
    content = path.read_bytes()
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError(f"Raw artifact changed: {reference['path']}")
    return content


def waveform_confirmation(reference, observed, audio, starts=None):
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    width = 4096
    starts = [15 * 48000, 35 * 48000, 55 * 48000] if starts is None else starts
    rows = []
    if len(reference) < width or not np.all(np.isfinite(reference)):
        return {"passed": False, "rows": [], "reason": "invalid_reference"}
    fft_size = 1 << (len(reference) + width - 2).bit_length()
    spectrum = np.fft.rfft(reference, fft_size)
    sums = np.concatenate(([0.0], np.cumsum(reference)))
    squares = np.concatenate(([0.0], np.cumsum(reference * reference)))
    variance = np.maximum(0, squares[width:] - squares[:-width] - (sums[width:] - sums[:-width]) ** 2 / width)
    for start in starts:
        row = {"observedStart": start, "referenceStart": None, "correlation": None,
               "alternativeCorrelation": None, "differenceFromFingerprintSamples": None, "passed": False}
        rows.append(row)
        target = observed[start:start + width]
        if len(target) != width or not np.all(np.isfinite(target)):
            row["reason"] = "missing_or_invalid_control_window"
            continue
        target = target - target.mean()
        energy = np.dot(target, target)
        if energy <= 0:
            row["reason"] = "silent_control_window"
            continue
        convolution = np.fft.irfft(spectrum * np.fft.rfft(target[::-1], fft_size), fft_size)
        numerator = convolution[width - 1:len(reference)]
        denominator = np.sqrt(variance * energy)
        scores = np.divide(numerator, denominator, out=np.full_like(numerator, -1.0), where=denominator > 0)
        best = int(np.argmax(scores))
        correlation = float(scores[best])
        scores[max(0, best - 24):best + 25] = -1
        alternative = float(np.max(scores))
        anchors = audio["anchors"]
        nearby = min(anchors, key=lambda anchor: abs(anchor["windowStart"] - start)) if anchors else None
        difference = best - (nearby["referenceStart"] - nearby["windowStart"] + start) if nearby else None
        row.update(referenceStart=best, correlation=correlation, alternativeCorrelation=alternative,
                   differenceFromFingerprintSamples=difference,
                   passed=bool(correlation >= 0.995 and alternative < 0.90 and difference is not None and abs(difference) <= 12))
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows,
            "method": "Global full-waveform NCC at prospectively fixed windows, independent of sign-key candidate selection",
            "scope": "Finite-fixture content correspondence control, not speaker/scanout or universal absolute latency"}


def scheduled_control(reference, observed, control):
    errors = []
    maximum = None
    checked = 0
    for marker in control["scheduled"]:
        for offset in range(256, marker["samples"] - 256, 128):
            checked += 1
            target_start = marker["referenceStart"] + offset
            target = reference[target_start:target_start + 256].astype(np.float64) * marker["gain"]
            center = marker["startFrame"] - control["firstFrame"] + offset
            energy = float(np.dot(target, target))
            candidates = []
            for shift in range(-24, 25):
                actual = observed[center + shift:center + shift + 256]
                if len(actual) == 256 and energy > 0 and np.sum((actual - target) ** 2) / energy <= 0.000001:
                    candidates.append(shift)
            if len(candidates) != 1 or abs(candidates[0]) > 12:
                errors.append({"scheduledFrame": marker["startFrame"] + offset, "matchingShifts": candidates})
            else:
                maximum = max(maximum or 0, abs(candidates[0]))
    return {"passed": checked == 180 and not errors and maximum is not None,
            "checkedWindows": checked, "maximumSampleError": maximum, "errors": errors,
            "scope": "Independently scheduled decoded-PCM buffers versus actual AudioWorklet sample positions"}


def foreground_interval(events, start, end):
    relevant = []
    for event in events:
        detail = event.get("detail")
        timestamp = event.get("performance")
        if isinstance(detail, dict) and ("visibility" in detail or "focused" in detail):
            if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
                return False
            relevant.append((timestamp, detail))
    if any(current[0] < previous[0] for previous, current in zip(relevant, relevant[1:])):
        return False
    initial = [detail for timestamp, detail in relevant if timestamp <= start]
    if not initial:
        return False
    return all(detail.get("visibility") == "visible" and detail.get("focused") is True
               for detail in [initial[-1], *[detail for timestamp, detail in relevant if start <= timestamp <= end]])


def analyze_envelope(envelope):
    result = envelope["result"]
    report = result.get("report")
    if not report or not result.get("audioPcm") or not result.get("controlAudioPcm"):
        return {"outcome": "UNRESOLVED", "summary": "Native recorder did not retain both PCM streams",
                "errors": result.get("errors", []), "nativeErrors": report.get("errors", []) if report else [],
                "candidateTiming": "NOT_RUN", "neural": "NOT_RUN"}
    media = result["media"]
    reference = np.frombuffer(artifact(media["pcm"]), dtype="<f4")
    observed = np.frombuffer(artifact(result["audioPcm"]), dtype="<f4")
    control_pcm = np.frombuffer(artifact(result["controlAudioPcm"]), dtype="<f4")
    audio_metadata = report["audio"]
    if len(observed) != audio_metadata["samples"] or len(control_pcm) != report["audioControl"]["samples"]:
        raise ValueError("PCM metadata sample count mismatch")
    audio = analyze_audio(reference, observed, observed_first_frame=audio_metadata["firstFrame"],
                          reference_first_media_time=media["validation"]["audio"]["firstSampleMediaTime"],
                          sample_rate=audio_metadata["sampleRate"])
    full_waveform = waveform_confirmation(reference, observed, audio)
    clock_control = scheduled_control(reference, control_pcm, report["audioControl"])
    frames = report["frames"]
    if not frames:
        return {"outcome": "UNRESOLVED", "summary": "No successful observed submissions", "clockControl": clock_control,
                "fullWaveform": full_waveform, "audio": audio, "candidateTiming": "NOT_RUN", "neural": "NOT_RUN"}
    collection_start = frames[0]["submitBefore"]
    timing = analyze_frames(frames, audio, fps=result["fps"], start_performance=collection_start,
                            end_performance=report["endPerformance"], completeness=report["completeness"],
                            timing_window=(report["startPerformance"], report["endPerformance"]))
    additional = {
        "nativeRecordedWithoutErrors": report["state"] == "RECORDED" and not report["errors"] and not result["errors"],
        "sampleClockControl": clock_control["passed"],
        "independentWaveformConfirmation": full_waveform["passed"],
        "contiguousAudio": not audio_metadata["overflow"] and not audio_metadata["discontinuity"],
        "foreground": bool(report["callbacks"]) and all(row["visibility"] == "visible" and row["focused"] for row in report["callbacks"])
        and foreground_interval(report["events"], collection_start, report["endPerformance"]),
        "exactNativeMetadataMatch": report["metadataMatchErrors"] == 0,
        "cleanup": report["cleanup"]["completed"] is True,
    }
    failed = [name for name, passed in {**timing["criteria"], **additional}.items() if not passed]
    return {"outcome": "FAIL" if failed else "PASS", "summary": {"failedCriteria": failed,
            "durationMs": timing["observation"]["durationMs"], "frames": timing["count"],
            "validCoverage": timing["validCoverage"], "halfWidthMs": timing["computedHalfWidthMs"],
            "signedPhaseMs": timing["absoluteUnsubtractedPhase"], "driftIntervalMs": timing["firstToLastDriftIntervalMs"]},
            "additionalCriteria": additional, "clockControl": clock_control, "fullWaveform": full_waveform,
            "audio": audio, "timing": timing, "candidateTiming": "NOT_RUN", "neural": "NOT_RUN",
            "semantics": "Finite-fixture conditional digital render-frontier calibration; no physical output bound",
            "completeCalibrationRequires": "All six prospectively ordered runs plus independent review and committed acceptance"}


if __name__ == "__main__":
    try:
        path = Path(sys.argv[1]).resolve()
        if not path.is_relative_to(ROOT / ".cache/m1010r"):
            raise ValueError("Calibration envelope outside ignored R cache")
        analysis = analyze_envelope(json.loads(path.read_text()))
        print(json.dumps(analysis, allow_nan=False, separators=(",", ":")))
    except Exception as error:
        print(json.dumps({"outcome": "UNRESOLVED", "summary": "Analysis input or instrument failure",
                          "error": str(error), "candidateTiming": "NOT_RUN", "neural": "NOT_RUN"}))