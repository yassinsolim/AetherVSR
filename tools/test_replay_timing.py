"""Finite digital controls, not native calibration or physical latency evidence."""

import numpy as np
import pytest

from m1010r.timing_analysis import analyze_audio, analyze_frames
from m1010r.analyze import waveform_confirmation, scheduled_control, analyze_envelope, foreground_interval


def test_foreground_blackout_between_callbacks_cannot_pass():
    events = [{"performance": 0, "detail": {"visibility": "visible", "focused": True}},
              {"performance": 30000, "detail": {"visibility": "hidden", "focused": False}},
              {"performance": 33000, "detail": {"visibility": "visible", "focused": True}}]
    assert not foreground_interval(events, 5000, 65000)
    assert foreground_interval(events, 34000, 65000)
    assert not foreground_interval([], 0, 60000)
    assert not foreground_interval(events, -1, 65000)
    assert not foreground_interval(events[::-1], 0, 65000)
    assert not foreground_interval([{"performance": 0, "detail": {"visibility": "visible"}}], 0, 60000)


def test_independent_waveform_confirmation_global_search_and_correspondence_bias():
    reference = np.random.default_rng(44).normal(size=20000).astype(np.float32)
    observed = reference[733:18000] * 0.5
    audio = analyze_audio(reference, observed, observed_first_frame=12000, reference_first_media_time=-0.05)
    control = waveform_confirmation(reference, observed, audio, starts=[1024, 8192])
    assert control["passed"]
    assert [row["referenceStart"] for row in control["rows"]] == [1757, 8925]
    for anchor in audio["anchors"]:
        anchor["referenceStart"] += 20
    assert not waveform_confirmation(reference, observed, audio, starts=[1024])["passed"]
    assert not waveform_confirmation(reference, observed * 0, audio, starts=[1024])["passed"]
    assert not waveform_confirmation(reference, observed, audio, starts=[19000])["passed"]


def test_independent_waveform_confirmation_rejects_duplicate_reference():
    fragment = np.random.default_rng(45).normal(size=8192).astype(np.float32)
    control = waveform_confirmation(np.tile(fragment, 3), fragment, {"anchors": []}, starts=[0])
    assert not control["passed"]
    assert control["rows"][0]["alternativeCorrelation"] > 0.99


@pytest.mark.parametrize("shift", [-13, -12, 0, 12, 13])
def test_scheduled_control_uses_independent_absolute_sample_positions(shift):
    reference = np.random.default_rng(46).normal(size=256000).astype(np.float32)
    observed = np.zeros(90000, dtype=np.float32)
    schedule = []
    for ordinal, gain in enumerate([1, 0.5, 0.75]):
        start = 12000 + ordinal * 24000
        reference_start = (ordinal + 1) * 48000
        observed[start + shift:start + shift + 8192] = reference[reference_start:reference_start + 8192] * gain
        schedule.append({"startFrame": start, "referenceStart": reference_start, "samples": 8192, "gain": gain})
    result = scheduled_control(reference, observed, {"firstFrame": 0, "scheduled": schedule})
    assert result["passed"] == (abs(shift) <= 12)
    assert result["checkedWindows"] == 180


def test_missing_native_recordings_cannot_qualify():
    result = analyze_envelope({"result": {"report": {"errors": ["stopped"]}, "errors": []}})
    assert result["outcome"] == "UNRESOLVED"
    assert result["neural"] == result["candidateTiming"] == "NOT_RUN"


def signal(length=8192):
    return np.random.default_rng(1010).uniform(-2048 / 32768, 2048 / 32768, length).astype(np.float32)


@pytest.mark.parametrize("shift", [-733, 517])
@pytest.mark.parametrize("gain", [0.1, 1, 2.75])
@pytest.mark.parametrize("pts_origin", [-0.021333333, 0.375])
def test_audio_shift_gain_and_signed_origins(shift, gain, pts_origin):
    reference = signal()
    observed = (reference[shift:] if shift > 0 else
                np.concatenate([-signal(-shift), reference])) * gain
    report = analyze_audio(reference, observed, observed_first_frame=90001,
                           reference_first_media_time=pts_origin)
    assert report["mappingValid"]
    assert report["referenceIndexedStarts"] == len(reference) - 511
    assert len(report["anchors"]) > 40
    for anchor in report["anchors"]:
        assert anchor["referenceStart"] == anchor["windowStart"] + shift
        assert anchor["renderSample"] == 90001 + anchor["windowStart"] + 256
        assert anchor["mediaTime"] == pytest.approx(pts_origin + (anchor["referenceStart"] + 256) / 48000)
        assert anchor["correlation"] >= 0.995
        assert abs(anchor["referenceStart"] - anchor["windowStart"] - shift) <= 12
    assert report["correspondenceUncertaintySamples"] == 12
    assert report["nativeCalibration"] == report["neural"] == "NOT_RUN"
    assert report["hardBoundStatus"] == "CONDITIONAL"
    assert not report["speakerOrScanoutBound"]


def test_audio_ambiguous_key_rejected_even_with_one_better_correlation():
    reference = signal(4096)
    reference[2048:2560:8] = reference[:512:8]
    report = analyze_audio(reference, reference[:1024], observed_first_frame=0,
                           reference_first_media_time=0)
    first = report["windows"][0]
    assert first["candidateCount"] == 2
    assert first["reasons"] == ["ambiguous_fingerprint"]
    assert first["referenceStart"] is None


def test_audio_held_out_noise_fails_full_window_correlation():
    reference = signal()
    observed = reference[:1024].copy()
    observed[1::8] *= -1
    report = analyze_audio(reference, observed, observed_first_frame=0,
                           reference_first_media_time=0)
    assert report["windows"][0]["candidateCount"] == 1
    assert report["windows"][0]["correlation"] < 0.995
    assert "correlation_below_threshold" in report["windows"][0]["reasons"]


@pytest.mark.parametrize("kind", ["missing", "silent", "nan", "duplicate"])
def test_audio_invalid_windows_retained(kind):
    reference = signal(2048)
    observed = signal(1024)
    expected = "missing_fingerprint"
    if kind == "missing":
        observed = -observed
    elif kind == "silent":
        observed[:] = 0
        expected = "zero_or_constant_observed_signal"
    elif kind == "nan":
        observed[7] = np.nan
        expected = "nonfinite_observed_window"
    else:
        reference = np.tile(reference, 2)
        expected = "ambiguous_fingerprint"
    report = analyze_audio(reference, observed, observed_first_frame=0,
                           reference_first_media_time=0)
    assert expected in report["windows"][0]["reasons"]
    assert report["invalidWindowCount"] > 0
    assert report["windows"][-1]["reasons"] == ["incomplete_window"]


@pytest.mark.parametrize("backward", [True, False])
def test_audio_mapping_discontinuity_invalidates_epoch(backward):
    reference = signal()
    observed = np.concatenate([reference[2048:4096], reference[:2048] if backward else reference[4608:6656]])
    report = analyze_audio(reference, observed, observed_first_frame=128,
                           reference_first_media_time=-0.2)
    assert not report["mappingValid"]
    assert report["anchors"] == []
    assert report["matchedWindowCount"] > 0
    assert any(issue["reason"] == ("non_monotonic_mapping" if backward else "not_constant_1x")
               for issue in report["mappingIssues"])


def test_audio_empty_and_invalid_origins():
    report = analyze_audio([], [], observed_first_frame=0, reference_first_media_time=0)
    assert not report["mappingValid"]
    assert report["matchedCorrelations"]["median"] is None
    assert report["empiricalOffsetResidualSamples"]["max"] is None
    for kwargs in ({"observed_first_frame": -1}, {"observed_first_frame": 1.25},
                   {"reference_first_media_time": np.nan}, {"sample_rate": 44100}):
        arguments = dict(observed_first_frame=0, reference_first_media_time=0)
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            analyze_audio([], [], **arguments)


@pytest.fixture
def audio():
    reference = signal(12000)
    return analyze_audio(reference, reference, observed_first_frame=48000,
                         reference_first_media_time=-0.02)


def frame(**changes):
    result = dict(audioBefore=1.1, audioAfter=1.1005, submitBefore=100,
                  submitAfter=101, readyAt=103, sourceIdentity=3, identityValid=True,
                  sequence=0, generation=1, mediaTime=0.1, presentedFrames=4,
                  presentationTime=99, expectedDisplayTime=116)
    result.update(changes)
    return result


def analyze(frames, audio):
    return analyze_frames(frames, audio, fps=30, start_performance=0, end_performance=60000)


def test_frame_enclosing_anchors_signed_interval_and_readiness(audio):
    report = analyze([frame(mediaTime=0.066)], audio)
    row = report["rows"][0]
    assert row["valid"]
    assert row["videoPts"] == 0.1
    assert row["raw"]["mediaTime"] == 0.066
    lower, upper = row["lowerAnchor"], row["upperAnchor"]
    assert lower["renderSample"] <= 1.1 * 48000
    assert upper["renderSample"] >= 1.1005 * 48000
    assert row["audioMediaInterval"] == [lower["mediaTime"] - 12 / 48000,
                                         upper["mediaTime"] + 12 / 48000]
    assert row["intervalMs"] == pytest.approx([(0.1 - upper["mediaTime"] - 12 / 48000) * 1000,
                                              (0.1 - lower["mediaTime"] + 12 / 48000) * 1000])
    assert row["intervalMs"][0] <= 20 <= row["intervalMs"][1]
    assert 0.25 <= row["halfWidthMs"] <= 10
    assert row["readiness"]["performanceInterval"] == [100, 103]
    assert row["readiness"]["elapsedUpperMs"] == 3
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("field", ["audioBefore", "audioAfter", "submitBefore", "submitAfter",
                                   "readyAt", "sourceIdentity", "identityValid", "sequence",
                                   "generation", "mediaTime", "presentedFrames", "presentationTime",
                                   "expectedDisplayTime"])
def test_frame_missing_field_preserves_row_and_counts(audio, field):
    record = frame()
    del record[field]
    report = analyze([record], audio)
    assert report["rows"][0]["raw"] == record
    assert f"missing_{field}" in report["rows"][0]["reasons"]
    assert report["validCount"] == 0 and report["invalidCount"] == 1
    assert report["validCoverage"] == 0
    assert report["absoluteUnsubtractedPhase"]["signedMidpointMs"]["median"] is None


@pytest.mark.parametrize("changes,reason", [
    ({"audioBefore": np.nan}, "nonfinite_audioBefore"),
    ({"readyAt": None}, "nonfinite_readyAt"),
    ({"sourceIdentity": -1}, "invalid_sourceIdentity"),
    ({"sourceIdentity": True}, "nonfinite_sourceIdentity"),
    ({"identityValid": False}, "invalid_source_identity"),
    ({"submitAfter": 99}, "unordered_submitBefore_submitAfter"),
    ({"audioAfter": 1.09}, "unordered_audioBefore_audioAfter"),
    ({"readyAt": 100}, "ready_before_submission"),
    ({"expectedDisplayTime": 98}, "unordered_presentation_metadata"),
    ({"submitAfter": 102.01}, "bracket_exceeds_2ms"),
    ({"audioAfter": 1.14}, "half_width_exceeds_10ms"),
    ({"audioBefore": 0.9}, "audio_extrapolation_forbidden"),
    ({"audioAfter": 1.3}, "audio_extrapolation_forbidden"),
    ({"submitBefore": -1}, "outside_observation"),
])
def test_frame_invalid_brackets_and_no_extrapolation(audio, changes, reason):
    report = analyze([frame(**changes)], audio)
    assert reason in report["rows"][0]["reasons"]
    assert not report["rows"][0]["valid"]
    assert not report["prospectiveCriteriaMet"]


def test_frame_cannot_bridge_unknown_audio_windows():
    reference = signal(12000)
    observed = reference.copy()
    observed[4800] = np.nan
    audio = analyze_audio(reference, observed, observed_first_frame=48000, reference_first_media_time=0)
    report = analyze([frame(audioBefore=1.09, audioAfter=1.12)], audio)
    assert audio["mappingValid"]
    assert report["rows"][0]["reasons"] == ["unknown_audio_gap"]
    assert report["rows"][0]["intervalMs"] is None


def test_frame_sequence_gaps_and_generation_are_not_hidden(audio):
    first = frame()
    second = frame(sequence=3, audioBefore=1.11, audioAfter=1.111, submitBefore=110, submitAfter=111, readyAt=114)
    report = analyze([first, second], audio)
    assert report["missingSequenceCount"] == 2
    assert report["validCoverage"] == 0.5
    assert report["coverageDenominator"] == 4
    second["generation"] = 2
    report = analyze([first, second], audio)
    assert report["validCount"] == 0
    assert all("mixed_generations" in row["reasons"] for row in report["rows"])


def test_frame_regressions_rejected(audio):
    report = analyze([frame(sequence=2), frame(sequence=1, audioBefore=1.09)], audio)
    assert "non_monotonic_sequence" in report["rows"][1]["reasons"]
    assert "non_monotonic_audioBefore" in report["rows"][1]["reasons"]
    assert "non_monotonic_submitBefore" in report["rows"][1]["reasons"]


def test_frame_empty_does_not_fabricate_zero(audio):
    report = analyze([], audio)
    assert report["validCoverage"] is None
    assert report["invalidOrMissingFraction"] is None
    assert report["bracketMs"]["max"] is None
    assert report["firstToLastDriftIntervalMs"] is None
    assert report["empiricalMidpointDriftMs"] is None
    assert report["hardBoundStatus"] == "UNAVAILABLE"
    assert not report["first20s"]["valid"]
    assert report["neural"] == report["nativeCalibration"] == "NOT_RUN"


def test_delayed_completion_is_not_an_audio_or_speaker_shift(audio):
    baseline = analyze([frame()], audio)
    delayed = analyze([frame(readyAt=500)], audio)
    assert baseline["rows"][0]["intervalMs"] == delayed["rows"][0]["intervalMs"]
    assert delayed["rows"][0]["readiness"]["elapsedUpperMs"] == 400
    assert not delayed["speakerOrScanoutBound"]
    assert delayed["hardBoundStatus"] == "CONDITIONAL"


@pytest.mark.parametrize("pts_shift", [-0.075, 0.043])
def test_known_audio_pts_shift_changes_signed_phase_by_opposite_amount(pts_shift):
    reference = signal(12000)
    base = analyze_audio(reference, reference, observed_first_frame=48000, reference_first_media_time=0)
    shifted = analyze_audio(reference, reference, observed_first_frame=48000, reference_first_media_time=pts_shift)
    base_row = analyze([frame()], base)["rows"][0]
    shifted_row = analyze([frame()], shifted)["rows"][0]
    assert np.subtract(shifted_row["intervalMs"], base_row["intervalMs"]) == pytest.approx([-pts_shift * 1000] * 2)
    assert shifted_row["intervalMs"][0] <= -pts_shift * 1000 <= shifted_row["intervalMs"][1]


@pytest.fixture(scope="module")
def long_audio():
    reference = signal(70 * 48000)
    for second in range(70):
        reference[second * 48000 + 12000:second * 48000 + 12048] += 0.5
    return analyze_audio(reference, reference[53017:] * np.float32(0.37),
                         observed_first_frame=96000, reference_first_media_time=-1024 / 48000)


def long_frames(fps=30, *, include_warmup=False):
    records = []
    for offset in range(-5 * fps if include_warmup else 0, 60 * fps):
        performance = 5000 + offset * 1000 / fps
        source_identity = 5 * fps + offset
        sequence = source_identity + 1 if include_warmup else offset
        records.append(frame(sequence=sequence, submitBefore=performance, submitAfter=performance + 1,
                             readyAt=performance + 3, sourceIdentity=source_identity,
                             audioBefore=2 + performance / 1000, audioAfter=2 + performance / 1000 + 0.0003,
                             mediaTime=source_identity / fps, presentedFrames=source_identity + 1,
                             presentationTime=performance - 1, expectedDisplayTime=performance + 1000 / fps))
    return records


def certificate(count=1800, first_sequence=0, **changes):
    result = dict(firstSequence=first_sequence, lastSequence=first_sequence + count - 1 if count else 0,
                  count=count, encodedCount=count, observedCallbackCount=count)
    result.update(changes)
    return result


def analyze_long(records, audio, fps=30, **options):
    options.setdefault("completeness", certificate(60 * fps))
    return analyze_frames(records, audio, fps=fps, start_performance=5000, end_performance=65000,
                          **options)


def test_missing_completeness_preserves_intervals_but_cannot_qualify(long_audio):
    report = analyze_frames(long_frames(), long_audio, fps=30,
                            start_performance=5000, end_performance=65000)
    assert report["validCoverage"] == 1
    assert all(row["intervalMs"] is not None for row in report["rows"])
    assert report["completeness"]["reasons"] == ["missing_completeness"]
    assert not report["criteria"]["collectionCompletenessValidated"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("edge", [0, -1])
def test_omitted_first_or_last_observation_cannot_qualify(long_audio, edge):
    records = long_frames()
    del records[edge]
    report = analyze_long(records, long_audio)
    assert report["coverageDenominator"] == 1800
    assert report["missingSequenceCount"] == 1
    assert report["validCoverage"] == pytest.approx(1799 / 1800)
    assert "collected_boundary_mismatch" in report["completeness"]["reasons"]
    assert not report["first20s" if edge == 0 else "last20s"]["valid"]
    assert not report["prospectiveCriteriaMet"]


def test_all_expected_observations_missing(audio):
    report = analyze_long([], audio)
    assert report["count"] == report["validCount"] == 0
    assert report["coverageDenominator"] == report["missingSequenceCount"] == 1800
    assert report["validCoverage"] == 0
    assert report["invalidOrMissingFraction"] == 1
    assert not report["completeness"]["validated"]
    assert not report["prospectiveCriteriaMet"]


def test_complete_collection_validated_before_warmup_selection(long_audio):
    records = long_frames(include_warmup=True)
    report = analyze_frames(records, long_audio, fps=30, start_performance=0, end_performance=65000,
                            completeness=certificate(1950, 1), timing_window=(5000, 65000))
    assert report["completeness"]["validated"]
    assert report["collectedCount"] == len(report["rows"]) == 1950
    assert report["excludedFromTimingCount"] == 150
    assert report["count"] == report["validCount"] == report["coverageDenominator"] == 1800
    assert not report["rows"][149]["inTimingWindow"]
    assert report["rows"][150]["inTimingWindow"]
    assert report["observation"]["startPerformance"] == 5000
    assert report["collection"]["startPerformance"] == 0
    assert report["prospectiveCriteriaMet"], report["criteria"]


@pytest.mark.parametrize("omitted", [0, 149, 150, -1])
def test_missing_warmup_or_timing_boundary_cannot_qualify(long_audio, omitted):
    records = long_frames(include_warmup=True)
    del records[omitted]
    report = analyze_frames(records, long_audio, fps=30, start_performance=0, end_performance=65000,
                            completeness=certificate(1950, 1), timing_window=(5000, 65000))
    assert report["completeness"]["missingSequenceCount"] == 1
    assert not report["completeness"]["validated"]
    assert not report["prospectiveCriteriaMet"]
    assert report["coverageDenominator"] == (1801 if omitted == 149 else 1800)
    if omitted == 0:
        assert report["validCoverage"] == 1


def test_prefiltered_rows_require_their_own_independent_certificate(long_audio):
    records = long_frames(include_warmup=True)[150:]
    whole = analyze_long(records, long_audio, completeness=certificate(1950, 1))
    assert not whole["completeness"]["validated"]
    assert not whole["prospectiveCriteriaMet"]
    window = analyze_long(records, long_audio, completeness=certificate(1800, 151))
    assert window["completeness"]["validated"]
    assert window["prospectiveCriteriaMet"], window["criteria"]


@pytest.mark.parametrize("field", ["firstSequence", "lastSequence", "count", "encodedCount", "observedCallbackCount"])
@pytest.mark.parametrize("bad_value", [-1, 0.5, True, np.bool_(True), None, np.nan, float("inf"),
                                      2**53, pytest.param(10**1000, id="overflow"), "1"])
def test_invalid_certificate_integers_cannot_validate(audio, field, bad_value):
    expected = certificate(1)
    expected[field] = bad_value
    report = analyze_long([frame()], audio, completeness=expected)
    assert f"invalid_completeness_{field}" in report["completeness"]["reasons"]
    assert report["completeness"]["expectedCount"] is None
    assert not report["completeness"]["validated"]
    assert not report["prospectiveCriteriaMet"]
    assert report["rows"][0]["intervalMs"] is not None


@pytest.mark.parametrize("field", ["firstSequence", "lastSequence", "count", "encodedCount", "observedCallbackCount"])
def test_missing_certificate_field_is_not_inferred(audio, field):
    expected = certificate(1)
    del expected[field]
    report = analyze_long([frame()], audio, completeness=expected)
    assert f"invalid_completeness_{field}" in report["completeness"]["reasons"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("bad_certificate", [False, 1, "complete", [], [("count", 1), ("count", 1)]])
def test_non_mapping_certificate_is_rejected(audio, bad_certificate):
    report = analyze_long([frame()], audio, completeness=bad_certificate)
    assert report["completeness"]["reasons"] == ["invalid_completeness"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("expected", [certificate(0, 1), certificate(0, lastSequence=1),
                                      certificate(2, 1, lastSequence=1), certificate(2, 2, lastSequence=1)])
def test_inconsistent_certificate_range_is_rejected(audio, expected):
    report = analyze_long([], audio, completeness=expected)
    assert "invalid_completeness_range" in report["completeness"]["reasons"]
    assert report["completeness"]["expectedCount"] is None
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("field,reason", [("encodedCount", "encoded_count_mismatch"),
                                         ("observedCallbackCount", "observed_callback_count_mismatch")])
@pytest.mark.parametrize("difference", [-1, 1])
def test_independent_encoding_and_callback_totals_must_match_rows(long_audio, field, reason, difference):
    report = analyze_long(long_frames(), long_audio, completeness=certificate(**{field: 1800 + difference}))
    assert report["validCoverage"] == 1
    assert report["completeness"]["reasons"] == [reason]
    assert all(value for name, value in report["criteria"].items() if name != "collectionCompletenessValidated")
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("index", [0, 900, -2])
def test_duplicate_replacing_missing_sequence_cannot_validate(long_audio, index):
    records = long_frames()
    records[index + 1]["sequence"] = records[index]["sequence"]
    report = analyze_long(records, long_audio)
    assert report["count"] == 1800
    assert report["completeness"]["missingSequenceCount"] == 1
    assert "duplicate_collected_sequence" in report["completeness"]["reasons"]
    assert "missing_collected_sequences" in report["completeness"]["reasons"]
    assert report["coverageDenominator"] == 1801
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("fault,reason", [("duplicate", "duplicate_collected_sequence"),
                                         ("invalid", "invalid_collected_sequence"),
                                         ("reordered", "unordered_collected_sequence")])
def test_warmup_sequence_faults_cannot_be_filtered_away(long_audio, fault, reason):
    records = long_frames(include_warmup=True)
    if fault == "duplicate":
        records[1]["sequence"] = records[0]["sequence"]
    elif fault == "invalid":
        records[1]["sequence"] = True
    else:
        records[1]["sequence"], records[2]["sequence"] = records[2]["sequence"], records[1]["sequence"]
    report = analyze_frames(records, long_audio, fps=30, start_performance=0, end_performance=65000,
                            completeness=certificate(1950, 1), timing_window=(5000, 65000))
    assert report["validCoverage"] == 1
    assert reason in report["completeness"]["reasons"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("sequence", [0, 1801])
def test_unexpected_sequence_with_matching_row_count_is_rejected(long_audio, sequence):
    records = long_frames()
    expected = certificate(1800, 1)
    for record in records:
        record["sequence"] += 1
    records[900]["sequence"] = sequence
    report = analyze_long(records, long_audio, completeness=expected)
    assert report["completeness"]["missingSequenceCount"] == 1
    assert "unexpected_collected_sequence" in report["completeness"]["reasons"]
    assert not report["prospectiveCriteriaMet"]


def test_certified_empty_collection_is_not_a_timing_pass(audio):
    report = analyze_long([], audio, completeness=certificate(0))
    assert report["completeness"]["validated"]
    assert report["coverageDenominator"] == 0
    assert report["validCoverage"] is None
    assert not report["prospectiveCriteriaMet"]
    unexpected = analyze_long([frame()], audio, completeness=certificate(0))
    assert "unexpected_collected_sequence" in unexpected["completeness"]["reasons"]
    assert not unexpected["prospectiveCriteriaMet"]


def test_maximum_exact_certificate_range_needs_no_sequence_expansion(audio):
    report = analyze_long([], audio, completeness=certificate(2**53 - 1))
    assert report["coverageDenominator"] == report["missingSequenceCount"] == 2**53 - 1
    assert len(report["sequenceGaps"]) == 1
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("timing_window", [False, {}, (), (0,), (0, 1, 2), (None, 65000),
                                          (5000, float("nan")), (4999, 65000), (5000, 65001),
                                          (65000, 5000), (5000, 5000)])
def test_invalid_timing_subset_arguments(audio, timing_window):
    with pytest.raises(ValueError, match="timing_window"):
        analyze_long([], audio, timing_window=timing_window)


def test_unlocated_row_remains_in_timing_denominator(long_audio):
    records = long_frames(include_warmup=True)
    del records[0]["submitBefore"]
    report = analyze_frames(records, long_audio, fps=30, start_performance=0, end_performance=65000,
                            completeness=certificate(1950, 1), timing_window=(5000, 65000))
    assert report["completeness"]["validated"]
    assert report["rows"][0]["inTimingWindow"]
    assert report["count"] == report["coverageDenominator"] == 1801
    assert not report["criteria"]["collectionBoundsValid"]
    assert not report["prospectiveCriteriaMet"]


def test_subset_requires_complete_submission_bracket_and_60_seconds(long_audio):
    report = analyze_frames(long_frames(), long_audio, fps=30, start_performance=5000, end_performance=65000,
                            completeness=certificate(), timing_window=(5000, 64967))
    assert "outside_timing_window" in report["rows"][-1]["reasons"]
    assert report["count"] == report["coverageDenominator"] == 1800
    assert not report["criteria"]["observationAtLeast60s"]
    assert not report["last20s"]["valid"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("fps", [30, 60])
def test_full_70_second_all_start_index_and_60_second_summary(long_audio, fps):
    report = analyze_long(long_frames(fps), long_audio, fps)
    assert long_audio["referenceIndexedStarts"] == 70 * 48000 - 511
    assert long_audio["mappingValid"]
    assert long_audio["matchedWindowCount"] > 25000
    assert all(anchor["referenceStart"] - anchor["windowStart"] == 53017 for anchor in long_audio["anchors"])
    assert report["prospectiveCriteriaMet"], report["criteria"]
    assert report["validCoverage"] == 1
    assert report["first20s"]["valid"] and report["last20s"]["valid"]
    phase = report["absoluteUnsubtractedPhase"]
    known_phase = -(53017 - 1024) / 48
    assert phase["signedLowerMs"]["median"] <= known_phase <= phase["signedUpperMs"]["median"]
    assert phase["signedLowerMs"]["min"] <= phase["signedLowerMs"]["p95"] < -1000
    assert phase["absoluteMagnitudeUpperMs"]["median"] > 1000
    assert report["firstToLastDriftIntervalMs"][0] <= 0 <= report["firstToLastDriftIntervalMs"][1]
    assert abs(report["empiricalMidpointDriftMs"]) <= 12 / 48
    assert report["hardBoundStatus"] == "CONDITIONAL"
    assert report["nativeCalibration"] == report["neural"] == "NOT_RUN"
    assert not report["speakerOrScanoutBound"]


@pytest.mark.parametrize("edge", [0, -1])
def test_invalid_first_or_last_20s_stops_acceptance(long_audio, edge):
    records = long_frames()
    records[edge]["identityValid"] = False
    report = analyze_long(records, long_audio)
    assert report["validCoverage"] > 0.99
    assert not report["first20s" if edge == 0 else "last20s"]["valid"]
    assert not report["prospectiveCriteriaMet"]


@pytest.mark.parametrize("fault", ["identity", "sequence_gap", "clock", "audio_coverage", "hidden_bracket"])
def test_isolated_middle_fault_cannot_hide_behind_99_percent(long_audio, fault):
    records = long_frames()
    middle = records[900]
    if fault == "identity":
        middle["identityValid"] = False
        failed_gate = "allPlannedVideoIdentitiesValid"
    elif fault == "sequence_gap":
        del records[900]
        failed_gate = "allPlannedVideoIdentitiesValid"
    elif fault == "clock":
        previous_time = records[899]["submitBefore"]
        middle.update(submitBefore=previous_time + 0.1, submitAfter=previous_time + 1.1, readyAt=previous_time + 3.1)
        failed_gate = "orderedClocksAndSequence"
    elif fault == "audio_coverage":
        middle.update(audioBefore=200, audioAfter=200.0003)
        failed_gate = "allConvertedAudioIntervalsCovered"
    else:
        middle["submitAfter"] = middle["submitBefore"] + 3
        del middle["mediaTime"]
        failed_gate = "bracketMaximumAtMost2ms"
    report = analyze_long(records, long_audio)
    assert report["validCoverage"] > 0.99
    assert not report["criteria"][failed_gate]
    assert not report["prospectiveCriteriaMet"]
    if fault == "hidden_bracket":
        assert report["bracketMs"]["max"] == 3


def test_insufficient_coverage_and_sampling_support(long_audio):
    records = long_frames()
    for record in records[800:850]:
        record["identityValid"] = False
    report = analyze_long(records, long_audio)
    assert not report["criteria"]["validCoverageAtLeast99Percent"]
    assert report["first20s"]["valid"] and report["last20s"]["valid"]
    sparse = analyze_long(records[::2], long_audio)
    assert not sparse["first20s"]["valid"] and not sparse["last20s"]["valid"]


def test_drift_summary_preserves_absolute_phase_and_sign(long_audio):
    records = long_frames()
    for record in records:
        if record["submitBefore"] >= 45000:
            record["sourceIdentity"] += 2
    report = analyze_long(records, long_audio)
    assert report["empiricalMidpointDriftMs"] == pytest.approx(2000 / 30, abs=0.25)
    assert report["firstToLastDriftIntervalMs"][0] <= 2000 / 30 <= report["firstToLastDriftIntervalMs"][1]
    assert not report["criteria"]["driftWithinOneSourceFrame"]
    assert report["absoluteUnsubtractedPhase"]["signedMidpointMs"]["median"] < -1000


def test_malformed_and_extreme_numeric_rows_are_retained(audio):
    report = analyze([None, frame(sourceIdentity=10**1000), frame(audioBefore=float("inf"))], audio)
    assert report["count"] == report["invalidCount"] == 3
    assert report["rows"][0]["reasons"] == ["invalid_frame_record"]
    assert "nonfinite_sourceIdentity" in report["rows"][1]["reasons"]
    assert report["rows"][1]["raw"]["sourceIdentity"] == 10**1000


def test_nonfinite_reference_and_missing_entire_reference():
    reference = signal(4096)
    observed = reference.copy()
    reference[1] = np.nan
    result = analyze_audio(reference, observed, observed_first_frame=0, reference_first_media_time=0)
    assert "nonfinite_reference_window" in result["windows"][0]["reasons"]
    missing = analyze_audio([], observed, observed_first_frame=0, reference_first_media_time=0)
    assert not missing["anchors"]
    assert missing["windows"][0]["reasons"] == ["missing_fingerprint"]
    report = analyze([frame()], missing)
    assert "audio_mapping_unavailable" in report["rows"][0]["reasons"]


@pytest.mark.parametrize("arguments", [dict(fps=0), dict(fps=float("nan")),
                                       dict(start_performance=100, end_performance=99),
                                       dict(end_performance=float("inf"))])
def test_invalid_observation_arguments(audio, arguments):
    options = dict(fps=30, start_performance=0, end_performance=60000)
    options.update(arguments)
    with pytest.raises(ValueError):
        analyze_frames([], audio, **options)


@pytest.mark.parametrize("fault", ["completion", "presentation", "audio_clock"])
def test_missing_field_cannot_hide_independent_order_failure(long_audio, fault):
    records = long_frames()
    middle = records[900]
    if fault == "completion":
        del middle["audioBefore"]
        middle["readyAt"] = middle["submitBefore"]
        reason = "ready_before_submission"
    elif fault == "presentation":
        del middle["audioBefore"]
        middle["expectedDisplayTime"] = middle["presentationTime"] - 1
        reason = "unordered_presentation_metadata"
    else:
        del middle["audioAfter"]
        middle["audioBefore"] = -1
        reason = "negative_audio_clock"
    report = analyze_long(records, long_audio)
    assert report["validCoverage"] > 0.99
    assert reason in report["rows"][900]["reasons"]
    assert not report["criteria"]["orderedClocksAndSequence"]
    assert not report["prospectiveCriteriaMet"]