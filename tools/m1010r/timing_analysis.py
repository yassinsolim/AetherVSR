"""Offline M10.10R timing, never speaker latency or display scanout.

PCM inputs are contiguous first-channel samples from one constant-1x epoch.
The reference origin must come from independently validated decoded audio PTS,
including edit/skip handling; observed_first_frame is actual worklet currentFrame.
The prospective 12-sample correspondence allowance is NOT established by an
exact match. Finite-fixture controls and native calibration remain prerequisites.
"""

from collections import Counter
from collections.abc import Mapping
from math import isfinite
from numbers import Real

import numpy as np


SAMPLE_RATE = 48000
WINDOW = 512
STRIDE = 128
UNCERTAINTY_SAMPLES = 12
MIN_CORRELATION = 0.995
SEMANTICS = "video frame-start PTS minus audio render-frontier media time"


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value):
    return _finite(value) and 0 <= value <= 2**53 - 1 and int(value) == value


def _stats(values):
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(min(values)), "max": float(max(values)),
    }


def _fingerprints(samples, stride):
    count = max(0, (len(samples) - WINDOW) // stride + 1)
    keys = np.zeros(count, dtype=np.uint64)
    for bit in range(64):
        signs = samples[bit * 8:bit * 8 + count * stride:stride] > 0
        keys |= signs.astype(np.uint64) << np.uint64(bit)
    return keys


def _correlation(reference, observed):
    reference_scale = np.max(np.abs(reference))
    observed_scale = np.max(np.abs(observed))
    if reference_scale == 0 or observed_scale == 0:
        return None
    reference = reference / reference_scale
    observed = observed / observed_scale
    reference = reference - reference.mean()
    observed = observed - observed.mean()
    denominator = np.linalg.norm(reference) * np.linalg.norm(observed)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(reference, observed) / denominator, -1, 1))


def analyze_audio(reference, observed, *, observed_first_frame,
                  reference_first_media_time, sample_rate=SAMPLE_RATE):
    """Return raw windows, unique anchors, origins and conditional-bound metadata.

    Every reference sample start is indexed, not just multiples of STRIDE. A
    duplicate sign key is rejected even when only one candidate would correlate.
    Full-window centered NCC also checks the 448 samples absent from the key.
    No fuzzy search, resampling, offset prior, interpolation or extrapolation.
    Missing windows split contiguous anchor segments; any sample-offset change
    among matches invalidates this entire declared constant-1x epoch.
    """
    if not _integer(sample_rate) or sample_rate != SAMPLE_RATE:
        raise ValueError("Only actual 48000 Hz PCM is supported; no resampling")
    if not _integer(observed_first_frame):
        raise ValueError("observed_first_frame must be an absolute nonnegative integer")
    if not _finite(reference_first_media_time):
        raise ValueError("reference_first_media_time must be finite (signed PTS allowed)")
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if reference.ndim != 1 or observed.ndim != 1:
        raise ValueError("PCM must be one-dimensional first-channel samples")
    if observed_first_frame + len(observed) + WINDOW > 2**53 - 1:
        raise ValueError("absolute sample positions exceed exact integer range")
    if not np.isfinite(reference_first_media_time + len(reference) / sample_rate):
        raise ValueError("reference media timeline overflows")

    keys = _fingerprints(reference, 1)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    native_keys = _fingerprints(observed, STRIDE)
    left = np.searchsorted(sorted_keys, native_keys, side="left")
    right = np.searchsorted(sorted_keys, native_keys, side="right")
    windows = []
    matches = []
    segment = -1
    previous_start = None
    for ordinal, start in enumerate(range(0, len(observed), STRIDE)):
        row = {
            "windowStart": start, "sampleCount": min(WINDOW, len(observed) - start),
            "renderSample": int(observed_first_frame + start + WINDOW // 2),
            "referenceStart": None, "mediaTime": None, "correlation": None,
            "fingerprint": None, "candidateCount": None, "segment": None,
            "matched": False, "valid": False, "reasons": [],
        }
        windows.append(row)
        if row["sampleCount"] != WINDOW:
            row["reasons"].append("incomplete_window")
            continue
        native = observed[start:start + WINDOW]
        row["fingerprint"] = f"{int(native_keys[ordinal]):016x}"
        candidates = int(right[ordinal] - left[ordinal])
        row["candidateCount"] = candidates
        if not np.all(np.isfinite(native)):
            row["reasons"].append("nonfinite_observed_window")
        elif not np.any(native) or np.ptp(native / max(np.max(np.abs(native)), 1)) == 0:
            row["reasons"].append("zero_or_constant_observed_signal")
        elif candidates == 0:
            row["reasons"].append("missing_fingerprint")
        elif candidates != 1:
            row["reasons"].append("ambiguous_fingerprint")
        else:
            reference_start = int(order[left[ordinal]])
            row["referenceStart"] = reference_start
            row["mediaTime"] = float(reference_first_media_time + (reference_start + WINDOW // 2) / sample_rate)
            target = reference[reference_start:reference_start + WINDOW]
            if not np.all(np.isfinite(target)):
                row["reasons"].append("nonfinite_reference_window")
            else:
                row["correlation"] = _correlation(target, native)
                if row["correlation"] is None:
                    row["reasons"].append("zero_or_constant_reference_signal")
                elif row["correlation"] < MIN_CORRELATION:
                    row["reasons"].append("correlation_below_threshold")
                else:
                    if previous_start is None or start != previous_start + STRIDE:
                        segment += 1
                    row.update(matched=True, valid=True, segment=segment)
                    previous_start = start
                    matches.append(row)

    issues = []
    for previous, current in zip(matches, matches[1:]):
        media_delta = current["referenceStart"] - previous["referenceStart"]
        render_delta = current["renderSample"] - previous["renderSample"]
        reason = ("non_monotonic_mapping" if media_delta <= 0 else
                  "not_constant_1x" if media_delta != render_delta else None)
        if reason:
            issues.append({"previousWindowStart": previous["windowStart"],
                           "windowStart": current["windowStart"], "reason": reason,
                           "referenceDeltaSamples": media_delta, "renderDeltaSamples": render_delta})
    mapping_valid = len(matches) >= 2 and not issues
    if not mapping_valid:
        for row in matches:
            row["valid"] = False
            row["reasons"].append("invalid_constant_1x_epoch" if issues else "insufficient_anchors")
    residuals = []
    if matches:
        origin_offset = matches[0]["referenceStart"] - matches[0]["windowStart"]
        residuals = [row["referenceStart"] - row["windowStart"] - origin_offset for row in matches]
    anchors = [{name: row[name] for name in (
        "windowStart", "renderSample", "referenceStart", "mediaTime", "correlation", "segment",
    )} for row in matches if row["valid"]]
    return {
        "sampleRate": int(sample_rate), "windowSamples": WINDOW, "strideSamples": STRIDE,
        "referenceSampleCount": len(reference), "observedSampleCount": len(observed),
        "referenceIndexedStarts": len(keys),
        "origins": {"observedFirstFrame": int(observed_first_frame),
                    "referenceFirstMediaTime": float(reference_first_media_time)},
        "windows": windows, "anchors": anchors, "mappingValid": mapping_valid,
        "mappingIssues": issues, "matchedWindowCount": len(matches),
        "validWindowCount": len(anchors), "invalidWindowCount": len(windows) - len(anchors),
        "invalidReasons": dict(Counter(reason for row in windows for reason in row["reasons"])),
        "matchedCorrelations": _stats([row["correlation"] for row in matches]),
        "empiricalOffsetResidualSamples": _stats(residuals),
        "residualSemantics": "relative to first matched offset; NOT absolute correspondence error",
        "correspondenceUncertaintySamples": UNCERTAINTY_SAMPLES,
        "correspondenceUncertaintyStatus": "PROSPECTIVE_NOT_UNIVERSAL",
        "hardBoundStatus": "CONDITIONAL" if mapping_valid else "UNAVAILABLE",
        "hardBoundConditions": ["validated finite-fixture correspondence within 12 samples",
                                "monotonic constant-1x epoch and contiguous recorded PCM",
                                "independently validated full decoded audio PTS origins",
                                "native calibration and independent acceptance review"],
        "nativeCalibration": "NOT_RUN", "neural": "NOT_RUN",
        "semantics": SEMANTICS, "speakerOrScanoutBound": False,
    }


FRAME_FIELDS = (
    "audioBefore", "audioAfter", "submitBefore", "submitAfter", "readyAt",
    "sourceIdentity", "sequence", "generation", "mediaTime", "presentedFrames",
    "presentationTime", "expectedDisplayTime",
)
INTEGER_FIELDS = ("sourceIdentity", "sequence", "generation", "presentedFrames")


def _phase_summary(rows):
    valid = [row for row in rows if row["valid"]]
    return {
        "signedLowerMs": _stats([row["intervalMs"][0] for row in valid]),
        "signedUpperMs": _stats([row["intervalMs"][1] for row in valid]),
        "signedMidpointMs": _stats([row["phaseMs"] for row in valid]),
        "halfWidthMs": _stats([row["halfWidthMs"] for row in valid]),
        "absoluteMagnitudeUpperMs": _stats([
            max(abs(row["intervalMs"][0]), abs(row["intervalMs"][1])) for row in valid
        ]),
    }


def _edge_window(rows, gaps, start, end, fps):
    selected = [row for row in rows if row["performanceTime"] is not None
                and start <= row["performanceTime"] <= end]
    valid = [row for row in selected if row["valid"]]
    ordered_times = sorted(row["performanceTime"] for row in selected)
    edge_allowance = 1000 / fps + 2
    distances = np.diff([start, *ordered_times, end]).tolist()
    gap_free = not any(gap["startPerformance"] is None or gap["endPerformance"] is None
                      or (gap["startPerformance"] <= end and gap["endPerformance"] >= start)
                      for gap in gaps)
    supported = bool(valid) and max(distances) <= edge_allowance
    unlocated = any(row["performanceTime"] is None for row in rows)
    return {
        "startPerformance": start, "endPerformance": end,
        "count": len(selected), "validCount": len(valid),
        "valid": bool(supported and gap_free and not unlocated and len(selected) == len(valid)),
        "maximumSamplingGapMs": float(max(distances)),
        "samplingGapLimitMs": edge_allowance,
        "phase": _phase_summary(selected),
    }


def _check_completeness(frames, certificate):
    reasons = []
    expected_count = None
    missing_count = None
    gaps = []
    fields = ("firstSequence", "lastSequence", "count", "encodedCount", "observedCallbackCount")
    if certificate is None:
        reasons.append("missing_completeness")
    elif not isinstance(certificate, Mapping):
        reasons.append("invalid_completeness")
    else:
        reasons.extend(f"invalid_completeness_{field}" for field in fields
                       if not _integer(certificate.get(field)))
    if not reasons:
        first, last, count = (int(certificate[field]) for field in fields[:3])
        if (count == 0 and (first != 0 or last != 0)) or (count > 0 and last - first + 1 != count):
            reasons.append("invalid_completeness_range")
        else:
            expected_count = count
        if certificate["encodedCount"] != count:
            reasons.append("encoded_count_mismatch")
        if certificate["observedCallbackCount"] != count:
            reasons.append("observed_callback_count_mismatch")
        if len(frames) != count:
            reasons.append("serialized_count_mismatch")
        sequences = [int(frame["sequence"]) for frame in frames
                     if isinstance(frame, Mapping) and _integer(frame.get("sequence"))]
        if len(sequences) != len(frames):
            reasons.append("invalid_collected_sequence")
        if len(set(sequences)) != len(sequences):
            reasons.append("duplicate_collected_sequence")
        if any(current <= previous for previous, current in zip(sequences, sequences[1:])):
            reasons.append("unordered_collected_sequence")
        if sequences and (count == 0 or min(sequences) != first or max(sequences) != last):
            reasons.append("collected_boundary_mismatch")
        if expected_count is not None:
            present = sorted({sequence for sequence in sequences if first <= sequence <= last}) if count else []
            missing_count = count - len(present)
            if any(count == 0 or not first <= sequence <= last for sequence in sequences):
                reasons.append("unexpected_collected_sequence")
            if missing_count:
                reasons.append("missing_collected_sequences")
                for previous, current in zip([first - 1, *present], [*present, last + 1]):
                    if current - previous > 1:
                        gaps.append({"afterSequence": previous if previous >= first else None,
                                     "beforeSequence": current if current <= last else None,
                                     "missingCount": current - previous - 1})
    return {"certificate": dict(certificate) if isinstance(certificate, Mapping) else certificate,
            "validated": not reasons, "reasons": reasons, "expectedCount": expected_count,
            "missingSequenceCount": missing_count, "sequenceGaps": gaps}


def analyze_frames(frames, audio, *, fps, start_performance, end_performance,
                   completeness=None, timing_window=None):
    """Enclose submission phase without interpolating the audio timeline.

    audioBefore/audioAfter are AudioContext seconds; all Performance timestamps
    and collection bounds are milliseconds on the same main clock. frames must
    contain every planned observation in one generation/constant-1x epoch, with
    source identities independently validated against frame-start container PTS.
    Raw rVFC mediaTime is retained, never substituted for sourceIdentity/fps.

    completeness must be independently recorded by the collector, never inferred
    from serialized rows: firstSequence, lastSequence, count, encodedCount and
    observedCallbackCount. All are nonnegative exact integers; every callback
    must be encoded and submitted once. Empty collections use zero boundaries.
    The certificate covers ALL supplied frames, including warmup, and is checked
    before selection. Without it, row/internal-gap coverage is descriptive only
    and prospectiveCriteriaMet is false, even when validCoverage is 1.

    start_performance/end_performance enclose the entire collection.
    timing_window=(start, end), in the same milliseconds, optionally selects a
    closed subset by submitBefore; otherwise it defaults to the collection.
    A selected submission must also finish within that subset. Unknown-time
    rows remain in the timing denominator. rows retains the full collection
    with inTimingWindow flags; count/validCount/invalidCount and timing summaries
    refer only to selected rows. The coverage denominator is selected rows plus
    missing sequences in intersecting gaps. Boundary, crossing and unlocated
    gaps are charged in full, never interpolated. Completeness failures outside
    the timing window still prohibit qualification. Pre-filtered input needs
    its own independently recorded certificate for that actual window.
    Edge windows conservatively require sampling support throughout each 20s
    (no gap larger than one source-frame period plus the 2ms bracket budget).
    No single-call result establishes native calibration or candidate fitness.
    """
    if not _finite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not (_finite(start_performance) and _finite(end_performance)
            and end_performance > start_performance):
        raise ValueError("observation bounds must be finite and ordered milliseconds")
    if timing_window is None:
        timing_start, timing_end = start_performance, end_performance
    elif (not isinstance(timing_window, (tuple, list)) or len(timing_window) != 2
          or not all(_finite(bound) for bound in timing_window)
          or not start_performance <= timing_window[0] < timing_window[1] <= end_performance):
        raise ValueError("timing_window must be an ordered pair within collection bounds")
    else:
        timing_start, timing_end = timing_window
    frames = list(frames)
    completeness_report = _check_completeness(frames, completeness)
    anchors = audio["anchors"]
    sample_rate = audio["sampleRate"]
    if sample_rate != SAMPLE_RATE or audio["correspondenceUncertaintySamples"] != UNCERTAINTY_SAMPLES:
        raise ValueError("audio must use the fixed 48000 Hz / 12-sample contract")
    render_samples = np.array([anchor["renderSample"] for anchor in anchors], dtype=np.float64)
    audio_available = audio["mappingValid"] is True and len(anchors) >= 2
    if audio_available:
        audio_available = bool(
            np.all(np.isfinite(render_samples)) and np.all(np.diff(render_samples) > 0)
            and all(_finite(anchor["mediaTime"]) and _finite(anchor["correlation"])
                    and anchor["correlation"] >= MIN_CORRELATION for anchor in anchors)
            and all(current["referenceStart"] - previous["referenceStart"]
                    == current["renderSample"] - previous["renderSample"]
                    and current["mediaTime"] > previous["mediaTime"]
                    for previous, current in zip(anchors, anchors[1:]))
        )
    generations = {frame["generation"] for frame in frames if isinstance(frame, Mapping)
                   and _integer(frame.get("generation"))}
    rows = []
    gaps = []
    previous = {}
    for ordinal, frame in enumerate(frames):
        row = {
            "rowIndex": ordinal, "raw": dict(frame) if isinstance(frame, Mapping) else frame,
            "valid": False, "reasons": [], "performanceTime": None, "inTimingWindow": True,
            "videoPts": None, "audioMediaInterval": None, "intervalMs": None,
            "phaseMs": None, "halfWidthMs": None, "bracketMs": None,
            "lowerAnchor": None, "upperAnchor": None, "readiness": None,
        }
        rows.append(row)
        reasons = row["reasons"]
        if not isinstance(frame, Mapping):
            reasons.append("invalid_frame_record")
            continue
        for field in FRAME_FIELDS:
            if field not in frame:
                reasons.append(f"missing_{field}")
            elif not _finite(frame[field]):
                reasons.append(f"nonfinite_{field}")
            elif field in INTEGER_FIELDS and not _integer(frame[field]):
                reasons.append(f"invalid_{field}")
        if "identityValid" not in frame:
            reasons.append("missing_identityValid")
        elif frame["identityValid"] is not True:
            reasons.append("invalid_source_identity")
        if len(generations) > 1:
            reasons.append("mixed_generations")
        if _finite(frame.get("submitBefore")):
            row["performanceTime"] = float(frame["submitBefore"])
            row["inTimingWindow"] = timing_start <= row["performanceTime"] <= timing_end
        if _integer(frame.get("sequence")):
            if "sequence" in previous:
                delta = frame["sequence"] - previous["sequence"]
                if delta <= 0:
                    reasons.append("non_monotonic_sequence")
                elif delta > 1:
                    gaps.append({"afterSequence": previous["sequence"],
                                 "beforeSequence": frame["sequence"], "missingCount": int(delta - 1),
                                 "startPerformance": previous.get("sequencePerformance"),
                                 "endPerformance": row["performanceTime"]})
            previous["sequence"] = frame["sequence"]
            previous["sequencePerformance"] = row["performanceTime"]
        for before, after in (("submitBefore", "submitAfter"), ("audioBefore", "audioAfter")):
            if _finite(frame.get(before)) and _finite(frame.get(after)):
                if frame[before] > frame[after]:
                    reasons.append(f"unordered_{before}_{after}")
                if after in previous and frame[before] < previous[after]:
                    reasons.append(f"non_monotonic_{before}")
                previous[after] = frame[after]
                if before == "submitBefore":
                    width = float(frame[after]) - float(frame[before])
                    if not isfinite(width):
                        reasons.append("bracket_overflow")
                    else:
                        row["bracketMs"] = width
                        if width > 2:
                            reasons.append("bracket_exceeds_2ms")
        for field in ("sourceIdentity", "mediaTime", "presentedFrames", "presentationTime"):
            if _finite(frame.get(field)):
                if field in previous and frame[field] < previous[field]:
                    reasons.append(f"non_monotonic_{field}")
                previous[field] = frame[field]
        if ((_finite(frame.get("submitBefore")) and frame["submitBefore"] < start_performance)
                or (_finite(frame.get("submitAfter")) and frame["submitAfter"] > end_performance)):
            reasons.append("outside_observation")
        if (timing_window is not None and row["inTimingWindow"]
                and _finite(frame.get("submitAfter")) and frame["submitAfter"] > timing_end):
            reasons.append("outside_timing_window")
        if _finite(frame.get("audioBefore")) and frame["audioBefore"] < 0:
            reasons.append("negative_audio_clock")
        if _finite(frame.get("readyAt")) and _finite(frame.get("submitAfter")):
            if frame["readyAt"] < frame["submitAfter"]:
                reasons.append("ready_before_submission")
            elif _finite(frame.get("submitBefore")):
                elapsed_upper = float(frame["readyAt"]) - float(frame["submitBefore"])
                if not isfinite(elapsed_upper):
                    reasons.append("readiness_overflow")
                else:
                    row["readiness"] = {
                        "performanceInterval": [float(frame["submitBefore"]), float(frame["readyAt"])],
                        "elapsedUpperMs": elapsed_upper,
                        "semantics": "queue completion notification upper bound; includes callback delivery, not GPU duration",
                    }
        if (_finite(frame.get("presentationTime")) and _finite(frame.get("expectedDisplayTime"))
                and frame["presentationTime"] > frame["expectedDisplayTime"]):
            reasons.append("unordered_presentation_metadata")
        if any(reason.startswith(("missing_", "nonfinite_", "invalid_sourceIdentity",
                                  "invalid_sequence", "invalid_generation", "invalid_presentedFrames"))
               for reason in reasons):
            continue
        if not audio_available:
            reasons.append("audio_mapping_unavailable")
            continue
        before_sample = frame["audioBefore"] * sample_rate
        after_sample = frame["audioAfter"] * sample_rate
        if not np.isfinite(before_sample) or not np.isfinite(after_sample):
            reasons.append("audio_sample_overflow")
            continue
        lower_index = int(np.searchsorted(render_samples, before_sample, side="right")) - 1
        upper_index = int(np.searchsorted(render_samples, after_sample, side="left"))
        if lower_index < 0 or upper_index >= len(anchors):
            reasons.append("audio_extrapolation_forbidden")
            continue
        if lower_index > upper_index or before_sample > after_sample:
            reasons.append("unordered_audio_bracket")
            continue
        lower, upper = anchors[lower_index], anchors[upper_index]
        row["lowerAnchor"], row["upperAnchor"] = dict(lower), dict(upper)
        if lower["segment"] != upper["segment"] or any(
            current["renderSample"] - previous_anchor["renderSample"] != STRIDE
            for previous_anchor, current in zip(anchors[lower_index:upper_index], anchors[lower_index + 1:upper_index + 1])
        ):
            reasons.append("unknown_audio_gap")
            continue
        audio_min = lower["mediaTime"] - UNCERTAINTY_SAMPLES / sample_rate
        audio_max = upper["mediaTime"] + UNCERTAINTY_SAMPLES / sample_rate
        video_pts = frame["sourceIdentity"] / fps
        interval = [(video_pts - audio_max) * 1000, (video_pts - audio_min) * 1000]
        if not all(np.isfinite(value) for value in [video_pts, audio_min, audio_max, *interval]):
            reasons.append("interval_overflow")
            continue
        row.update(videoPts=float(video_pts), audioMediaInterval=[audio_min, audio_max],
                   intervalMs=interval, phaseMs=interval[0] / 2 + interval[1] / 2,
                   halfWidthMs=(audio_max - audio_min) * 500)
        if row["halfWidthMs"] > 10:
            reasons.append("half_width_exceeds_10ms")
        row["valid"] = not reasons

    collected_rows = rows
    if completeness_report["expectedCount"] is not None:
        sequence_times = {int(row["raw"]["sequence"]): row["performanceTime"] for row in rows
                          if isinstance(row["raw"], Mapping) and _integer(row["raw"].get("sequence"))}
        gaps = []
        for missing in completeness_report["sequenceGaps"]:
            gap_start = (start_performance if missing["afterSequence"] is None
                         else sequence_times.get(missing["afterSequence"]))
            gap_end = (end_performance if missing["beforeSequence"] is None
                       else sequence_times.get(missing["beforeSequence"]))
            if gap_start is not None and gap_end is not None and gap_start > gap_end:
                gap_start = gap_end = None
            gaps.append({**missing, "startPerformance": gap_start, "endPerformance": gap_end})
    rows = [row for row in collected_rows if row["inTimingWindow"]]
    timing_gaps = [gap for gap in gaps if gap["startPerformance"] is None or gap["endPerformance"] is None
                   or (gap["startPerformance"] <= timing_end and gap["endPerformance"] >= timing_start)]
    valid_rows = [row for row in rows if row["valid"]]
    missing_sequences = sum(gap["missingCount"] for gap in timing_gaps)
    denominator = len(rows) + missing_sequences
    coverage = len(valid_rows) / denominator if denominator else None
    first = _edge_window(rows, timing_gaps, timing_start, timing_start + 20000, fps)
    last = _edge_window(rows, timing_gaps, timing_end - 20000, timing_end, fps)
    if timing_end - timing_start < 40000:
        first["valid"] = last["valid"] = False
    drift_interval = None
    drift_midpoint = None
    if first["valid"] and last["valid"]:
        first_phase, last_phase = first["phase"], last["phase"]
        drift_interval = [last_phase["signedLowerMs"]["median"] - first_phase["signedUpperMs"]["median"],
                          last_phase["signedUpperMs"]["median"] - first_phase["signedLowerMs"]["median"]]
        drift_midpoint = last_phase["signedMidpointMs"]["median"] - first_phase["signedMidpointMs"]["median"]
    brackets = [row["bracketMs"] for row in rows if row["bracketMs"] is not None]
    widths = [row["halfWidthMs"] for row in rows if row["halfWidthMs"] is not None]
    converted = [row for row in rows if isinstance(row["raw"], Mapping)
                 and _finite(row["raw"].get("audioBefore")) and _finite(row["raw"].get("audioAfter"))
                 and row["raw"]["audioBefore"] <= row["raw"]["audioAfter"]]
    criteria = {
        "collectionCompletenessValidated": completeness_report["validated"],
        "collectionBoundsValid": bool(collected_rows) and all(
            row["performanceTime"] is not None and isinstance(row["raw"], Mapping)
            and _finite(row["raw"].get("submitAfter"))
            and start_performance <= row["performanceTime"] <= row["raw"]["submitAfter"] <= end_performance
            for row in collected_rows),
        "observationAtLeast60s": timing_end - timing_start >= 60000,
        "audioMappingValid": audio_available,
        "allPlannedVideoIdentitiesValid": bool(rows) and missing_sequences == 0
        and all(isinstance(row["raw"], Mapping) and row["raw"].get("identityValid") is True
                and _integer(row["raw"].get("sourceIdentity")) for row in rows),
        "allConvertedAudioIntervalsCovered": bool(converted)
        and all(row["audioMediaInterval"] is not None for row in converted),
        "orderedClocksAndSequence": bool(rows) and not any(
            reason.startswith(("unordered_", "non_monotonic_"))
            or reason in ("ready_before_submission", "negative_audio_clock", "bracket_overflow", "readiness_overflow")
            for row in collected_rows for reason in row["reasons"]),
        "bracketMaximumAtMost2ms": bool(brackets) and min(brackets) >= 0 and max(brackets) <= 2,
        "halfWidthMaximumAtMost10ms": bool(widths) and max(widths) <= 10,
        "validCoverageAtLeast99Percent": coverage is not None and coverage >= 0.99,
        "first20sValid": first["valid"], "last20sValid": last["valid"],
        "singleGeneration": len(generations) == 1,
        "driftWithinOneSourceFrame": drift_interval is not None
        and max(abs(value) for value in drift_interval) <= 1000 / fps,
    }
    return {
        "completeness": completeness_report,
        "rows": collected_rows, "collectedCount": len(collected_rows),
        "excludedFromTimingCount": len(collected_rows) - len(rows),
        "count": len(rows), "validCount": len(valid_rows),
        "invalidCount": len(rows) - len(valid_rows), "missingSequenceCount": missing_sequences,
        "sequenceGaps": gaps, "coverageDenominator": denominator, "validCoverage": coverage,
        "invalidOrMissingFraction": 1 - coverage if coverage is not None else None,
        "invalidReasons": dict(Counter(reason for row in rows for reason in row["reasons"])),
        "collection": {"startPerformance": float(start_performance), "endPerformance": float(end_performance)},
        "observation": {"fps": float(fps), "startPerformance": float(timing_start),
                "endPerformance": float(timing_end), "durationMs": float(timing_end - timing_start)},
        "coverageDenominatorSemantics": "timing rows (including unlocated rows) plus entire missing-sequence gaps intersecting timing window",
        "absoluteUnsubtractedPhase": _phase_summary(rows),
        "bracketMs": _stats(brackets), "computedHalfWidthMs": _stats(widths),
        "first20s": first, "last20s": last,
        "firstToLastDriftIntervalMs": drift_interval, "empiricalMidpointDriftMs": drift_midpoint,
        "empiricalDriftSemantics": "midpoint difference, not a hard-bound residual or control subtraction",
        "criteria": criteria, "prospectiveCriteriaMet": all(criteria.values()),
        "hardBoundStatus": "CONDITIONAL" if valid_rows else "UNAVAILABLE",
        "hardBoundConditions": [*audio["hardBoundConditions"],
                                "offline validated frame identity to frame-start PTS",
                                "complete planned observations and ordered same-clock brackets"],
        "correspondenceUncertaintySamples": UNCERTAINTY_SAMPLES,
        "matchedCorrelations": _stats([anchor["correlation"] for row in valid_rows
                                       for anchor in (row["lowerAnchor"], row["upperAnchor"])]),
        "nativeCalibration": "NOT_RUN", "neural": "NOT_RUN",
        "semantics": SEMANTICS, "speakerOrScanoutBound": False,
    }