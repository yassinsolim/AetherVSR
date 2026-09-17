"""Offline RI checks; no product or physical qualification."""

import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from . import analyze as legacy
    from .timing_analysis import analyze_audio, analyze_frames, _finite, _integer, _stats, _phase_summary
else:
    import analyze as legacy
    from timing_analysis import analyze_audio, analyze_frames, _finite, _integer, _stats, _phase_summary


RATE = 48000
FROZEN_MEDIA = {
    30: (1076460, "e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b",
         "b8eae0c174c6c5280f42bb9def3709c852201e0f4d0378aca6217a120b7655e4"),
    60: (1211083, "5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29",
         "4870cbc926a2dd56dc9b602ac42eebd7fbf58ea95c72cdf0ff33544b439a69aa"),
}
FROZEN_AUDIO = {
    "sampleRate": RATE, "decodedSamples": 3360000, "decodedPcmBytes": 13440000,
    "decodedPcmSha256": "146670bf67d516149ced77dc32b860ff9d7863e89f88743202e11d2b289a0f65",
    "firstPts": 0, "timeBase": "1/48000", "firstSampleMediaTime": 0,
    "endExclusiveMediaTime": 70, "decodedFrames": 3282,
    "ptsFramesSha256": "b9e0dfa7c80debb479b9172af2ec9941ab92d7b3168d4dc12760c364274b3c94",
}


def _object(value):
    return value if isinstance(value, dict) else {}


def _matches_fields(value, expected):
    value = _object(value)
    return all(value.get(key) == item and
        (value.get(key) is item if isinstance(item, bool) else
         _integer(value.get(key)) if isinstance(item, int) else True)
        for key, item in expected.items())


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not _finite(value):
        return None
    return value


def analyze_render(render, pcm, *, required_start, target_end, tail_frames):
    render = _object(render)
    terminal = _object(render.get("terminal"))
    errors = []
    for field in ("watchdogFired", "timedOut"):
        if render.get(field) is not False:
            errors.append(field)
    if not isinstance(render.get("errors"), list) or render["errors"]:
        errors.append("host_errors")
    if not terminal:
        errors.append("missing_terminal")
    if terminal.get("completionReason") != "RENDER_TARGET_REACHED":
        errors.append("non_success_terminal")
    for field in ("overflow", "discontinuity"):
        if terminal.get(field) is not False:
            errors.append(field)
    if terminal.get("sampleRate") != RATE:
        errors.append("sample_rate")
    fields = ("firstFrame", "actualObservedEndFrame", "requestedEndFrame", "processedSamples",
              "processedBlocks", "heartbeatCount")
    for field in fields:
        if not _integer(terminal.get(field)):
            errors.append(f"invalid_{field}")
    geometry = all(_integer(value) for value in (required_start, target_end, tail_frames))
    if not geometry or not required_start < target_end or tail_frames <= 0:
        errors.append("invalid_required_range")
    blocks = terminal.get("blockLengths")
    blocks_valid = isinstance(blocks, list) and bool(blocks) and all(
        _integer(length) and 0 < length <= 2**32 - 1 for length in blocks)
    if not blocks_valid:
        errors.append("invalid_block_lengths")
    prefix = [0]
    if blocks_valid:
        for length in blocks:
            prefix.append(prefix[-1] + int(length))
    maximum = max(blocks) if blocks_valid else None
    first, end = terminal.get("firstFrame"), terminal.get("actualObservedEndFrame")
    ranges_valid = _integer(first) and _integer(end) and end >= first
    requested = terminal.get("requestedEndFrame")
    overshoot = end - requested if _integer(end) and _integer(requested) else None
    tail = end - (target_end - tail_frames) if geometry and _integer(end) else None
    if blocks_valid:
        if len(blocks) != terminal.get("processedBlocks") or prefix[-1] != terminal.get("processedSamples"):
            errors.append("block_count_or_sum")
        if prefix[-1] > 80 * RATE:
            errors.append("storage_capacity")
    if not ranges_valid or end - first != terminal.get("processedSamples"):
        errors.append("noncontiguous_range")
    if not (geometry and ranges_valid and first <= required_start and end >= target_end):
        errors.append("required_range_not_covered")
    if not geometry or not _integer(requested) or requested != target_end:
        errors.append("requested_target_mismatch")
    if not blocks_valid or overshoot is None or not 0 <= overshoot < blocks[-1]:
        errors.append("target_not_in_final_whole_block")
    if tail is None or tail < tail_frames:
        errors.append("insufficient_tail")
    pcm_count = None if pcm is None else int(pcm.size)
    pcm_finite = None if pcm is None else bool(np.all(np.isfinite(pcm)))
    if pcm is None:
        errors.append("missing_pcm")
    elif pcm.ndim != 1 or not pcm_finite or pcm_count != terminal.get("processedSamples"):
        errors.append("pcm_shape_finiteness_or_count")

    heartbeats = render.get("heartbeats")
    if not isinstance(heartbeats, list):
        errors.append("missing_heartbeat_history")
        heartbeats = []
    if terminal.get("heartbeatCount") != len(heartbeats):
        errors.append("heartbeat_count")
    expected_blocks = []
    previous_end = None
    if blocks_valid:
        for index, cumulative in enumerate(prefix[1:-1], 1):
            if previous_end is None or cumulative - previous_end >= 12000:
                expected_blocks.append(index)
                previous_end = cumulative
    if [_object(beat).get("processedBlocks") for beat in heartbeats] != expected_blocks:
        errors.append("heartbeat_render_cadence")
    clock_rows = []
    progression = []
    previous_end = None
    for ordinal, value in enumerate(heartbeats, 1):
        beat = _object(value)
        integer_fields = ("heartbeatOrdinal", "currentFrame", "actualObservedEndFrame", "processedSamples",
                          "processedBlocks", "blockLength", "sampleRate")
        valid = all(_integer(beat.get(field)) for field in integer_fields)
        count = beat.get("processedBlocks")
        if (not valid or beat.get("heartbeatOrdinal") != ordinal or beat.get("sampleRate") != RATE
                or beat.get("state") != "RECORDING" or not blocks_valid or not ranges_valid
                or not 1 <= count <= len(blocks)):
            errors.append("heartbeat_metadata")
        else:
            beat_end = beat["actualObservedEndFrame"]
            if (beat["processedSamples"] != prefix[count] or beat_end != first + prefix[count]
                or beat["currentFrame"] != first + prefix[count - 1]
                or beat["blockLength"] != blocks[count - 1] or beat_end > end):
                errors.append("heartbeat_prefix")
            if previous_end is not None:
                progression.append(beat_end - previous_end)
                if beat_end - previous_end < 12000:
                    errors.append("heartbeat_progression")
            previous_end = beat_end
        clock_rows.append((beat, beat.get("actualObservedEndFrame")))
    observation = render.get("terminalObservation")
    if not isinstance(observation, dict):
        errors.append("missing_terminal_observation")
    else:
        clock_rows.append((observation, end))
    events = render.get("contextEvents")
    if not isinstance(events, list) or not events:
        errors.append("missing_context_history")
        events = []
    if any(_object(event).get("state") != "running" for event in events):
        errors.append("unexpected_context_state")

    residuals = []
    guard = None if maximum is None else 2 * maximum / RATE
    for row, end_frame in clock_rows:
        if _integer(end_frame) and all(_finite(row.get(field)) for field in ("contextBefore", "contextAfter")):
            lower = row["contextBefore"] - end_frame / RATE
            upper = row["contextAfter"] - end_frame / RATE
            if _finite(lower) and _finite(upper):
                residuals.append({"endFrame": end_frame, "lowerSeconds": lower, "upperSeconds": upper})
                if guard is None or lower < -guard - 1e-12:
                    errors.append("negative_clock_residual")
            else:
                errors.append("nonfinite_clock_residual")
    clock_fields = ("hostBefore", "hostAfter", "contextBefore", "contextAfter")
    ordered_events = []
    for history in ([row for row, _ in clock_rows], events):
        previous_host = previous_context = -1
        for value in history:
            row = _object(value)
            if not all(_finite(row.get(field)) and row[field] >= 0 for field in clock_fields):
                errors.append("invalid_receive_clock")
                continue
            if (row["hostAfter"] < row["hostBefore"] or row["contextAfter"] < row["contextBefore"]
                or row["hostBefore"] < previous_host or row["contextBefore"] < previous_context):
                errors.append("unordered_receive_clock")
            previous_host, previous_context = row["hostAfter"], row["contextAfter"]
            ordered_events.append(row)
    ordered_events.sort(key=lambda row: row["hostBefore"])
    for previous, current in zip(ordered_events, ordered_events[1:]):
        if current["hostBefore"] < previous["hostAfter"] or current["contextBefore"] < previous["contextAfter"]:
            errors.append("context_clock_regression")
    if events and isinstance(observation, dict) and _finite(observation.get("hostAfter")):
        if any(_finite(_object(event).get("hostAfter")) and event["hostAfter"] > observation["hostAfter"] for event in events):
            errors.append("context_after_terminal")
    return _safe({
        "passed": not errors, "errors": sorted(set(errors)), "raw": render,
        "completionReason": terminal.get("completionReason"),
        "range": {"firstFrame": first, "actualObservedEndFrame": end, "requestedEndFrame": requested,
            "processedSamples": terminal.get("processedSamples"), "processedBlocks": terminal.get("processedBlocks"),
            "blockLengthSum": prefix[-1] if blocks_valid else None, "maximumBlockLength": maximum,
            "overshootFrames": overshoot, "tailFrames": tail},
        "pcm": {"samples": pcm_count, "finite": pcm_finite},
        "heartbeat": {"count": len(heartbeats), "expectedCount": len(expected_blocks) if blocks_valid else None,
            "renderFrameDeltas": _stats(progression), "deliveryRateHz": None},
        "clock": {"residuals": residuals, "lowerSeconds": _stats([row["lowerSeconds"] for row in residuals]),
            "upperSeconds": _stats([row["upperSeconds"] for row in residuals]),
            "negativeGuardSeconds": guard, "positiveDeliveryBoundSeconds": None,
            "semantics": "Positive residual: delivery/rendering, not clock error or physical latency"}})


def _schedule(epoch):
    return [{"startFrame": epoch + 12000 + offset, "referenceStart": reference_start, "samples": 8192, "gain": gain}
        for offset, reference_start, gain in ((0, 48000, 1), (26400, 96000, 0.5), (45600, 144000, 0.75))]


def analyze_control(control, reference, observed):
    control = _object(control)
    epoch = control.get("epochFrame")
    epoch_valid = _integer(epoch) and epoch + 70592 <= 2**53 - 1
    scheduled = _schedule(epoch) if epoch_valid else None
    render = analyze_render(control.get("render"), observed,
        required_start=epoch + 12000 if epoch_valid else None, target_end=epoch + 70592 if epoch_valid else None, tail_frames=4800)
    errors = list(render["errors"])
    if (not epoch_valid or control.get("scheduled") != scheduled
            or control.get("targetEndFrame") != epoch + 70592 or control.get("postSignalTailFrames") != 4800
            or any(not _finite(region.get("gain")) for region in control["scheduled"])):
        errors.append("control_geometry")
    terminal = _object(_object(control.get("render")).get("terminal"))
    for field, terminal_field in (("firstFrame", "firstFrame"), ("samples", "processedSamples"), ("sampleRate", "sampleRate")):
        if not _integer(control.get(field)) or control[field] != terminal.get(terminal_field):
            errors.append(f"control_{field}_mismatch")
    windows = {"passed": False, "checkedWindows": 0, "maximumSampleError": None,
        "errors": [], "reason": "missing_reference_or_observed_pcm"}
    if reference is not None and observed is not None and scheduled is not None and _integer(terminal.get("firstFrame")):
        if len(reference) < 256000 or not np.all(np.isfinite(reference)):
            errors.append("invalid_control_reference")
        else:
            windows = legacy.scheduled_control(reference, observed, {"firstFrame": terminal["firstFrame"], "scheduled": scheduled})
    verified = windows["checkedWindows"] - len(windows["errors"])
    if not windows["passed"]:
        errors.append("scheduled_windows_not_verified")
    if (not _integer(control.get("expectedWindows")) or control["expectedWindows"] != 180
            or not _integer(control.get("verifiedWindows")) or control["verifiedWindows"] != verified
            or control.get("maximumSampleError") != windows["maximumSampleError"]
            or (control.get("maximumSampleError") is not None and not _integer(control["maximumSampleError"]))):
        errors.append("control_window_metadata")
    if not isinstance(control.get("errors"), list) or control["errors"]:
        errors.append("native_control_errors")
    return _safe({"passed": not errors, "errors": sorted(set(errors)), "render": render, "windows": windows,
        "expectedWindows": 180, "verifiedWindows": verified,
        "maximumSampleError": windows["maximumSampleError"], "raw": control})


def _window_valid(window):
    epoch = window.get("epochFrame")
    return (_integer(epoch) and epoch + 3216000 <= 2**53 - 1 and window == {
        "epochFrame": epoch, "startFrame": epoch + 240000, "endFrame": epoch + 3120000,
        "targetEndFrame": epoch + 3216000, "sampleRate": RATE})


def analyze_timing(report, audio, *, fps, maximum_block, video_validation):
    frames = report.get("frames")
    window = _object(report.get("renderWindow"))
    if not isinstance(frames, list) or not frames or not _window_valid(window) or fps not in (30, 60):
        return {"passed": False, "errors": ["missing_frames_or_invalid_render_window"],
                "rows": _safe(frames), "uncertaintyBoundMs": None}
    collection_start = _object(frames[0]).get("submitBefore")
    collection_end = report.get("endPerformance")
    if not (_finite(collection_start) and _finite(collection_end) and 0 <= collection_start < collection_end):
        return {"passed": False, "errors": ["invalid_original_collection_bounds"],
                "rows": _safe(frames), "uncertaintyBoundMs": None}
    original = analyze_frames(frames, audio, fps=fps, start_performance=collection_start,
        end_performance=collection_end, completeness=report.get("completeness"))
    start, end = window["startFrame"] / RATE, window["endFrame"] / RATE
    rows = original["rows"]
    unlocated = 0
    block_ms = maximum_block * 1000 / RATE if _integer(maximum_block) and maximum_block > 0 else None
    tick_ms = 1000 / 15360 if video_validation.get("timeBase") == "1/15360" and video_validation.get("allIdentitiesAndPtsExact") is True else None
    for row in rows:
        raw = _object(row["raw"])
        before, after = raw.get("audioBefore"), raw.get("audioAfter")
        located = _finite(before) and _finite(after) and 0 <= before <= after
        row["inTimingWindow"] = not located or (before >= start and after < end)
        row["audioWindowLocated"] = bool(located)
        unlocated += not located
        base = row["intervalMs"]
        bracket = row["bracketMs"]
        row["baseIntervalMs"] = base
        components = {
            "anchorEnclosureWidthMs": None if base is None else base[1] - base[0],
            "anchorSpacingMs": None if base is None else (row["upperAnchor"]["mediaTime"] - row["lowerAnchor"]["mediaTime"]) * 1000,
            "correspondenceHalfWidthMsAlreadyIncluded": 12 * 1000 / RATE,
            "maximumBlockMs": block_ms, "sampleQuantizationMs": 1000 / RATE,
            "mainClockBracketMs": bracket, "videoPtsTickMs": tick_ms}
        row["uncertaintyComponents"] = components
        row["reasons"] = [reason for reason in row["reasons"] if reason != "half_width_exceeds_10ms"]
        row["intervalMs"] = row["halfWidthMs"] = row["widthMs"] = row["phaseMs"] = None
        if not located:
            row["reasons"].append("unlocated_audio_window")
        if base is not None and all(value is not None and _finite(value) and value >= 0 for value in components.values()):
            expansion = block_ms + 1000 / RATE + bracket + tick_ms
            row["intervalMs"] = [base[0] - expansion, base[1] + expansion]
            row["widthMs"] = row["intervalMs"][1] - row["intervalMs"][0]
            row["halfWidthMs"] = row["widthMs"] / 2
            row["phaseMs"] = sum(row["intervalMs"]) / 2
            if row["halfWidthMs"] > 15:
                row["reasons"].append("half_width_exceeds_15ms")
        else:
            row["reasons"].append("missing_uncertainty_component")
        row["valid"] = not row["reasons"]
    selected = [row for row in rows if row["inTimingWindow"]]
    sequence_rows = {row["raw"]["sequence"]: _object(row["raw"]) for row in rows
        if _integer(_object(row["raw"]).get("sequence"))}
    gaps = []
    for gap in original["sequenceGaps"]:
        low = sequence_rows.get(gap["afterSequence"], {}).get("audioAfter")
        high = sequence_rows.get(gap["beforeSequence"], {}).get("audioBefore")
        located = _finite(low) and _finite(high) and low <= high
        gaps.append({"startAudio": low if located else None, "endAudio": high if located else None,
                     "missingCount": gap["missingCount"]})

    def intersects(gap, lower, upper):
        return gap["startAudio"] is None or gap["endAudio"] >= lower and gap["startAudio"] < upper

    def edge(lower, upper):
        subset = [row for row in selected if row["audioWindowLocated"]
            and lower <= row["raw"]["audioBefore"] and row["raw"]["audioAfter"] < upper]
        times = sorted(row["raw"]["audioBefore"] for row in subset)
        maximum_gap = max(np.diff([lower, *times, upper])) * 1000
        valid = bool(subset) and all(row["valid"] for row in subset)
        valid = valid and not unlocated and not any(intersects(gap, lower, upper) for gap in gaps)
        return {"startAudio": lower, "endAudio": upper, "count": len(subset),
                "validCount": sum(row["valid"] for row in subset),
                "valid": bool(valid and maximum_gap <= 1000 / fps + 2 + 1e-9),
                "maximumSamplingGapMs": float(maximum_gap), "samplingGapLimitMs": 1000 / fps + 2,
                "phase": _phase_summary(subset)}

    first, last = edge(start, start + 20), edge(end - 20, end)
    drift = midpoint_drift = None
    if first["valid"] and last["valid"]:
        drift = [last["phase"]["signedLowerMs"]["median"] - first["phase"]["signedUpperMs"]["median"],
                 last["phase"]["signedUpperMs"]["median"] - first["phase"]["signedLowerMs"]["median"]]
        midpoint_drift = last["phase"]["signedMidpointMs"]["median"] - first["phase"]["signedMidpointMs"]["median"]
    missing = sum(gap["missingCount"] for gap in gaps if intersects(gap, start, end))
    denominator = len(selected) + missing
    valid_count = sum(row["valid"] for row in selected)
    coverage = valid_count / denominator if denominator else None
    widths = [row["halfWidthMs"] for row in selected if row["halfWidthMs"] is not None]
    bound = max(widths) if selected and len(widths) == len(selected) and not missing and not unlocated else None
    reuse = ("collectionCompletenessValidated", "collectionBoundsValid", "audioMappingValid",
             "orderedClocksAndSequence", "singleGeneration", "allPlannedVideoIdentitiesValid")
    criteria = {name: original["criteria"][name] for name in reuse}
    criteria["fixtureIdentityRange"] = all(_integer(_object(row["raw"]).get("sourceIdentity"))
        and row["raw"]["sourceIdentity"] < 70 * fps for row in rows)
    criteria.update(
        audioWindowLocated=unlocated == 0, fixedAudioWindow=_window_valid(window),
        allConvertedAudioIntervalsCovered=bool(selected) and all(row["audioMediaInterval"] is not None for row in selected),
        bracketMaximumAtMost2ms=bool(selected) and all(row["bracketMs"] is not None and 0 <= row["bracketMs"] <= 2 for row in selected),
        halfWidthMaximumAtMost15ms=bound is not None and bound <= 15,
        validCoverageAtLeast99Percent=coverage is not None and coverage >= 0.99,
        first20sValid=first["valid"], last20sValid=last["valid"],
        driftWithinOneSourceFrame=drift is not None and max(abs(value) for value in drift) <= 1000 / fps)
    phase = _phase_summary(selected)
    return _safe({
        "passed": all(criteria.values()), "errors": [name for name, passed in criteria.items() if not passed],
        "criteria": criteria, "rows": rows, "completeness": original["completeness"],
        "collection": original["collection"], "renderWindow": window, "durationAudioSeconds": 60,
        "collectedCount": len(rows), "count": len(selected), "excludedFromTimingCount": len(rows) - len(selected),
        "unlocatedCount": unlocated, "validCount": valid_count, "invalidCount": len(selected) - valid_count,
        "missingSequenceCount": missing, "sequenceGaps": gaps, "coverageDenominator": denominator,
        "validCoverage": coverage, "computedHalfWidthMs": _stats(widths), "uncertaintyBoundMs": bound,
        "absoluteUnsubtractedPhase": phase, "medianDigitalPhaseMs": phase["signedMidpointMs"]["median"],
        "first20s": first, "last20s": last, "firstToLastDriftIntervalMs": drift,
        "empiricalMidpointDriftMs": midpoint_drift,
        "uncertaintySemantics": "Both sides expanded; anchor allowances counted once"})


def analyze_envelope(envelope):
    result = _object(_object(envelope).get("result"))
    report = _object(result.get("report"))
    media = _object(result.get("media"))
    failures, missing = [], []
    verified = []

    def load(reference, name, pcm=False):
        if reference is None:
            missing.append(f"missing_{name}")
            return None
        try:
            content = legacy.artifact(reference)
            verified.append({"name": name, **reference})
            if not pcm:
                return content
            samples = np.frombuffer(content, dtype="<f4")
            if not np.all(np.isfinite(samples)):
                failures.append(f"nonfinite_{name}")
            return samples
        except FileNotFoundError:
            missing.append(f"missing_{name}")
            return None
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{name}: {error}")
            return None

    control_pcm = load(result.get("controlAudioPcm"), "control_pcm", True)
    observed = load(result.get("audioPcm"), "media_pcm", True)
    reference = load(media.get("pcm"), "reference_pcm", True)
    load(media.get("media"), "media_asset")
    if result.get("rawSnapshot") is not None:
        load(result["rawSnapshot"], "raw_snapshot")
    snapshot = load(result.get("finalRawSnapshot"), "final_raw_snapshot")
    original_report = dict(report)
    for key, ref in (("audioControl", result.get("controlAudioPcm")), ("audio", result.get("audioPcm"))):
        alias = _object(report.get(key)).get("pcm")
        if alias is None and ref is not None:
            missing.append(f"missing_{key}_pcm_reference")
        elif alias is not None and alias != ref:
            load(alias, key)
            failures.append(f"{key}_pcm_reference_mismatch")
        if isinstance(report.get(key), dict):
            original_report[key] = {field: value for field, value in report[key].items() if field != "pcm"}
    if snapshot is not None:
        try:
            if json.loads(snapshot) != original_report:
                failures.append("final_raw_snapshot_mismatch")
        except (ValueError, UnicodeError, TypeError):
            failures.append("invalid_final_raw_snapshot")
    control = analyze_control(report.get("audioControl"), reference, control_pcm)
    if report.get("audioControl") is None:
        missing.append("missing_control_report")
    elif not control["passed"]:
        failures.append("short_control_failed")
    window = _object(report.get("renderWindow"))
    media_render = analyze_render(report.get("renderAudio"), observed, required_start=window.get("startFrame"),
        target_end=window.get("targetEndFrame"), tail_frames=96000)
    if report.get("renderAudio") is None:
        missing.append("missing_media_render")
    elif not media_render["passed"]:
        failures.append("media_render_failed")
    if report.get("renderWindow") is None:
        missing.append("missing_render_window")
    elif not _window_valid(window):
        failures.append("invalid_render_window")
    if report.get("instrument") != "M10.10RI":
        failures.append("not_M10.10RI")
    options = _object(report.get("options"))
    frames = report.get("frames", [])
    if (result.get("fps") not in (30, 60) or options.get("fps", result.get("fps")) != result.get("fps")
            or not isinstance(frames, list) or any(_object(row).get("neural") is not False for row in frames)
            or any(key in result for key in ("candidate", "candidateTiming"))):
        failures.append("not_baseline_instrument_run")
    for source, name in ((result, "runner"), (report, "native")):
        if not isinstance(source.get("errors"), list) or source["errors"]:
            failures.append(f"{name}_errors")
    if report.get("state") != "RECORDED":
        missing.append("native_not_recorded")
    cleanup = _object(report.get("cleanup"))
    cleanup_ok = (cleanup.get("completed") is True and result.get("pageClosed") is True
        and cleanup.get("audioContext") == "closed" and cleanup.get("videoPaused") is True
        and all(_integer(cleanup.get(field)) and cleanup[field] == 0 for field in
            ("pipeline", "probe", "device", "audioNodes", "observers", "timers", "pendingCompletions")))
    if not cleanup_ok:
        missing.append("cleanup_not_verified")
    if not _integer(report.get("metadataMatchErrors")) or report["metadataMatchErrors"] != 0:
        failures.append("native_metadata_mismatch")
    validation = _object(media.get("validation"))
    video_validation, audio_validation = _object(validation.get("video")), _object(validation.get("audio"))
    fps = result.get("fps")
    audio = full_waveform = timing = None
    if media.get("pcm") is not None:
        if (not _matches_fields(media["pcm"], {"bytes": FROZEN_AUDIO["decodedPcmBytes"],
            "sha256": FROZEN_AUDIO["decodedPcmSha256"]})
            or not _matches_fields(audio_validation, FROZEN_AUDIO)
            or reference is not None and reference.size != FROZEN_AUDIO["decodedSamples"]):
            failures.append("reference_audio_validation")
    if media:
        frozen = FROZEN_MEDIA.get(fps) if _integer(fps) else None
        if frozen is None or not _matches_fields(video_validation, {
            "allIdentitiesAndPtsExact": True, "fps": fps, "timeBase": "1/15360", "width": 1280,
            "height": 720, "frames": 70 * fps, "rowsSha256": frozen[2]}):
            failures.append("reference_video_validation")
        if media.get("media") is not None and (frozen is None or not _matches_fields(
            media["media"], {"bytes": frozen[0], "sha256": frozen[1]})):
            failures.append("reference_media_identity")
    terminal = _object(_object(report.get("renderAudio")).get("terminal"))
    metadata = _object(report.get("audio"))
    if report.get("audio") is None:
        missing.append("missing_audio_metadata")
    else:
        expected = {"type": "audio-recording", "sampleRate": RATE, "overflow": False, "discontinuity": False,
            "encoding": "Float32 little-endian PCM", "channel": 0}
        if (not _matches_fields(metadata, expected)
                or any(not _integer(metadata.get(field)) or metadata[field] != terminal.get(terminal_field)
                    for field, terminal_field in (("firstFrame", "firstFrame"), ("samples", "processedSamples"),
                                                  ("sampleRate", "sampleRate")))
                or any(metadata.get(field) is not terminal.get(field) for field in ("overflow", "discontinuity"))
                or not isinstance(metadata.get("blockLengths"), list)
                or not all(_integer(length) and length > 0 for length in metadata["blockLengths"])
                or metadata["blockLengths"] != terminal.get("blockLengths")
                or not _integer(metadata.get("bytes")) or metadata["bytes"] != metadata["samples"] * 4
                or observed is not None and metadata["bytes"] != observed.nbytes):
            failures.append("audio_metadata_mismatch")
    if reference is not None and observed is not None:
        try:
            audio = analyze_audio(reference, observed, observed_first_frame=terminal.get("firstFrame"),
                reference_first_media_time=audio_validation.get("firstSampleMediaTime"), sample_rate=terminal.get("sampleRate"))
            if not audio["mappingValid"] or audio["invalidReasons"].get("ambiguous_fingerprint", 0):
                failures.append("ambiguous_or_invalid_audio_mapping")
            full_waveform = legacy.waveform_confirmation(reference, observed, audio)
            if not full_waveform["passed"]:
                failures.append("global_waveform_confirmation")
            timing = analyze_timing(report, audio, fps=fps, maximum_block=media_render["range"]["maximumBlockLength"],
                video_validation=video_validation)
            if not timing["passed"]:
                failures.append("timing_failed")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            failures.append(f"media_analysis: {error}")
    callbacks = report.get("callbacks")
    foreground = False
    if isinstance(callbacks, list) and callbacks and timing is not None and "collection" in timing:
        bounds = timing["collection"]
        try:
            foreground = (all(isinstance(row, dict) and row.get("visibility") == "visible" and row.get("focused") is True
                and _integer(row.get("readyState")) and 2 <= row["readyState"] <= 4 for row in callbacks)
                and isinstance(report.get("events"), list)
                and legacy.foreground_interval(report["events"], bounds["startPerformance"], bounds["endPerformance"]))
        except (TypeError, AttributeError):
            foreground = False
    if not foreground:
        (failures if callbacks else missing).append("foreground_not_verified")
    if timing is None:
        missing.append("timing_unavailable")
    outcome = "FAIL" if failures else "UNRESOLVED" if missing else "PASS"
    bound = timing.get("uncertaintyBoundMs") if outcome == "PASS" and timing else None
    return _safe({
        "outcome": outcome, "instrument": "M10.10RI", "scope": "Conditional digital instrument",
        "summary": {"reason": "; ".join(sorted(set(failures + missing))) or "Per-run checks passed",
            "failedCriteria": sorted(set(failures)), "missingEvidence": sorted(set(missing)),
            "controlVerifiedWindows": control["verifiedWindows"], "controlMaximumSampleError": control["maximumSampleError"],
            "medianDigitalPhaseMs": timing.get("medianDigitalPhaseMs") if timing else None, "uncertaintyBoundMs": bound},
        "control": control, "mediaRender": media_render, "audio": audio, "fullWaveform": full_waveform,
        "timing": timing, "foreground": foreground, "cleanup": cleanup_ok,
        "verifiedArtifacts": verified, "nativeErrors": report.get("errors"), "runnerErrors": result.get("errors"),
        "uncertaintyBoundMs": bound, "physicalLatencyBoundMs": None,
        "completeQualificationRequires": "Native 30/60/30/60; per-FPS median spread <=1 frame; independent review"})


def main():
    try:
        path = Path(sys.argv[1]).resolve()
        if not path.is_relative_to(legacy.ROOT / ".cache/m1010r"):
            raise ValueError("Envelope outside R cache")
        analysis = analyze_envelope(json.loads(path.read_text()))
    except Exception as error:
        analysis = {"outcome": "UNRESOLVED", "summary": {"reason": f"Analysis input failure: {error}"},
            "instrument": "M10.10RI", "uncertaintyBoundMs": None, "physicalLatencyBoundMs": None}
    print(json.dumps(_safe(analysis), allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()