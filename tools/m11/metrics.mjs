const finite = value => typeof value === 'number' && Number.isFinite(value);
const ordinal = value => Number.isSafeInteger(value) && value >= 0;
const inWindow = (at, start, stop) => finite(at) && at >= start && at < stop;
const difference = (end, start) => finite(end) && finite(start) && finite(end - start) ? end - start : null;
const maxOrNull = values => values.length ? Math.max(...values) : null;
const FRAME_QUANTIZATION_MS = 0.001;
const RUNTIME_STATES = new Set(['warmup', 'stable', 'fallback', 'probing', 'manual-baseline',
  'suspended', 'unavailable', 'failed']);

function distribution(values) {
  const sorted = values.filter(finite).sort((left, right) => left - right);
  const percentile = fraction => {
    if (!sorted.length) return null;
    const position = (sorted.length - 1) * fraction, lower = Math.floor(position);
    return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
  };
  return { count: sorted.length, missing: values.length - sorted.length,
    min: sorted[0] ?? null, p50: percentile(0.5), p95: percentile(0.95),
    max: sorted.at(-1) ?? null, values };
}

function latestAt(sorted, at, field) {
  let lower = 0, upper = sorted.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (sorted[middle][field] <= at) lower = middle + 1;
    else upper = middle;
  }
  return sorted[lower - 1] ?? null;
}

export function analyzePlayback(record, { fps, requireNeural, minDurationMs = 60000 } = {}) {
  if (fps !== 30 && fps !== 60) throw new RangeError('fps must be 30 or 60');
  if (typeof requireNeural !== 'boolean') throw new TypeError('requireNeural must be boolean');
  if (!finite(minDurationMs) || minDurationMs < 60000) throw new RangeError('minDurationMs must be at least 60000');
  const period = 1000 / fps, minimumFps = fps === 60 ? 58 : 29;
  const startAt = record?.startAt, stopAt = record?.stopAt;
  const intervalValid = finite(startAt) && finite(stopAt) && startAt >= 0 && stopAt > startAt;
  const durationMs = intervalValid ? stopAt - startAt : null;
  const rows = Array.isArray(record?.rows) ? record.rows : [];
  const samples = Array.isArray(record?.samples) ? record.samples : [];
  const states = Array.isArray(record?.states) ? record.states : [];
  const selected = intervalValid ? rows.filter(row => inWindow(row?.submittedAt, startAt, stopAt)) : [];
  const selectedStates = intervalValid ? states.filter(state => inWindow(state?.at, startAt, stopAt)) : [];
  const timedStates = [...selectedStates].sort((left, right) => left.at - right.at);
  const freshnessStates = intervalValid ? states.filter(state => finite(state?.at) && state.at >= startAt && state.at <= stopAt)
    .sort((left, right) => left.at - right.at) : [];
  const ordered = (intervalValid ? rows.filter(row => finite(row?.submittedAt) && row.submittedAt >= startAt && row.submittedAt <= stopAt) : [])
    .map((row, index) => ({ ...row, order: index }))
    .sort((left, right) => left.submittedAt - right.submittedAt || left.order - right.order);
  const counts = { inputRows: rows.length, observations: selected.length,
    outsideWindow: rows.filter(row => finite(row?.submittedAt)).length - selected.length,
    unboundedRows: rows.filter(row => !finite(row?.submittedAt)).length,
    usefulSubmissions: 0, renderedOpportunities: 0, invalidObservations: 0,
    missingIdentities: 0, missingMetadata: 0, misalignedIdentities: 0,
    duplicateSequences: 0, sequenceRegressions: 0, duplicateIdentities: 0, identityRegressions: 0,
    generationRegressions: 0, staleGenerations: 0, stalePresentations: 0, clockErrors: 0,
    missingOpportunities: 0, afterStopOpportunities: 0, superseded: 0, hidden: 0,
    nonNeuralSubmissions: 0, submissionSequenceGaps: 0, sourceIdentityGapFrames: 0,
    mediaPtsGapFrames: 0 };
  const seenSequences = new Set(), identitiesByGeneration = new Map();
  let lastSequence = null, lastGeneration = null, lastBefore = null, lastSubmit = null;
  let lastMedia = null, lastMediaGeneration = null;
  const observations = selected.map(row => {
    const identityKnown = row.identityValid === true && ordinal(row.sourceIdentity);
    const metadataKnown = finite(row.mediaTime) && row.mediaTime >= 0;
    const rawAge = identityKnown && metadataKnown ? (row.mediaTime - row.sourceIdentity / fps) * 1000 : null;
    const sourceCallbackAgeMs = finite(rawAge) ? rawAge : null;
    const aligned = finite(sourceCallbackAgeMs) && Math.abs(sourceCallbackAgeMs) <= period + FRAME_QUANTIZATION_MS;
    const bracket = difference(row.submittedAt, row.before);
    const rawOpportunityDelay = difference(row.opportunityAt, row.submittedAt);
    const boundedOpportunity = finite(row.opportunityAt) && row.opportunityAt >= row.submittedAt && row.opportunityAt <= stopAt;
    const opportunityDelay = boundedOpportunity ? rawOpportunityDelay : null;
    const elapsedToOpportunity = boundedOpportunity ? difference(row.opportunityAt, row.before) : null;
    const visualAge = finite(sourceCallbackAgeMs) && finite(elapsedToOpportunity) ? sourceCallbackAgeMs + elapsedToOpportunity : null;
    const validSequence = ordinal(row.sequence) && row.sequence > 0;
    const duplicateSequence = validSequence && seenSequences.has(row.sequence);
    const sequenceRegressed = validSequence && lastSequence !== null && row.sequence < lastSequence;
    if (validSequence) {
      if (lastSequence !== null && row.sequence > lastSequence) counts.submissionSequenceGaps += row.sequence - lastSequence - 1;
      seenSequences.add(row.sequence); lastSequence = Math.max(lastSequence ?? 0, row.sequence);
    }
    const validGeneration = ordinal(row.generation);
    const generationRegressed = validGeneration && lastGeneration !== null && row.generation < lastGeneration;
    if (validGeneration) lastGeneration = Math.max(lastGeneration ?? 0, row.generation);
    const priorIdentity = identitiesByGeneration.get(row.generation);
    const duplicateIdentity = identityKnown && priorIdentity?.seen.has(row.sourceIdentity) === true;
    const identityRegressed = identityKnown && priorIdentity !== undefined && row.sourceIdentity < priorIdentity.maximum;
    if (identityKnown && validGeneration) {
      if (priorIdentity) {
        if (row.sourceIdentity > priorIdentity.maximum) counts.sourceIdentityGapFrames += row.sourceIdentity - priorIdentity.maximum - 1;
        priorIdentity.seen.add(row.sourceIdentity); priorIdentity.maximum = Math.max(priorIdentity.maximum, row.sourceIdentity);
      } else identitiesByGeneration.set(row.generation, { seen: new Set([row.sourceIdentity]), maximum: row.sourceIdentity });
    }
    const mediaRegressed = metadataKnown && lastMediaGeneration === row.generation && lastMedia !== null && row.mediaTime < lastMedia;
    if (metadataKnown) {
      if (lastMediaGeneration === row.generation && lastMedia !== null && row.mediaTime > lastMedia) {
        counts.mediaPtsGapFrames += Math.max(0, Math.round((row.mediaTime - lastMedia) * fps) - 1);
      }
      lastMedia = row.mediaTime; lastMediaGeneration = row.generation;
    }
    const clockValid = finite(bracket) && bracket >= 0 && row.before >= 0
      && (lastBefore === null || row.before >= lastBefore) && (lastSubmit === null || row.submittedAt >= lastSubmit)
      && (!finite(row.opportunityAt) || row.opportunityAt >= row.submittedAt) && !mediaRegressed;
    const callbackIntervalMs = difference(row.before, lastBefore);
    if (finite(row.before)) lastBefore = row.before;
    lastSubmit = row.submittedAt;
    const submitState = latestAt(timedStates, row.submittedAt, 'at');
    const opportunityState = boundedOpportunity ? latestAt(freshnessStates, row.opportunityAt, 'at') : null;
    const staleGeneration = !validGeneration || generationRegressed
      || (submitState !== null && submitState.generation > row.generation);
    const latest = boundedOpportunity ? latestAt(ordered, row.opportunityAt, 'submittedAt') : null;
    const latestMatches = latest?.sequence === row.sequence && latest?.generation === row.generation
      && latest?.sourceIdentity === row.sourceIdentity;
    const sameOpportunityGeneration = opportunityState === null || opportunityState.generation <= row.generation;
    const visible = row.canvasVisible === true && row.visibility === 'visible';
    const stalePresentation = boundedOpportunity && visible && row.superseded === false
      && (!latestMatches || !sameOpportunityGeneration || staleGeneration || sequenceRegressed || identityRegressed);
    const usefulSubmission = validSequence && !duplicateSequence && !sequenceRegressed && !staleGeneration
      && !duplicateIdentity && !identityRegressed && clockValid && aligned && (!requireNeural || row.neural === true);
    const rendered = usefulSubmission && boundedOpportunity && visible && row.superseded === false
      && latestMatches && sameOpportunityGeneration;
    counts.usefulSubmissions += Number(usefulSubmission); counts.renderedOpportunities += Number(rendered);
    counts.invalidObservations += Number(!rendered);
    counts.missingIdentities += Number(!identityKnown); counts.missingMetadata += Number(!metadataKnown);
    counts.misalignedIdentities += Number(identityKnown && metadataKnown && !aligned);
    counts.duplicateSequences += Number(duplicateSequence); counts.sequenceRegressions += Number(sequenceRegressed || !validSequence);
    counts.duplicateIdentities += Number(duplicateIdentity); counts.identityRegressions += Number(identityRegressed);
    counts.generationRegressions += Number(generationRegressed); counts.staleGenerations += Number(staleGeneration);
    counts.stalePresentations += Number(stalePresentation); counts.clockErrors += Number(!clockValid);
    counts.missingOpportunities += Number(!finite(row.opportunityAt)); counts.afterStopOpportunities += Number(finite(row.opportunityAt) && row.opportunityAt > stopAt);
    counts.superseded += Number(row.superseded === true || (boundedOpportunity && !latestMatches));
    counts.hidden += Number(!visible); counts.nonNeuralSubmissions += Number(row.neural !== true);
    return { sequence: row.sequence ?? null, generation: row.generation ?? null, submittedAt: row.submittedAt,
      opportunityAt: finite(row.opportunityAt) ? row.opportunityAt : null, usefulSubmission, rendered,
      sourceCallbackAgeMs, encodeSubmissionMs: bracket, rawOpportunityDelayMs: rawOpportunityDelay,
      opportunityDelayMs: opportunityDelay, elapsedToOpportunityMs: elapsedToOpportunity,
      opportunityVisualAgeMs: visualAge, callbackLatencyMs: difference(row.before, row.presentationTime), callbackIntervalMs };
  });
  const validSamples = [], seenSamples = new Set();
  const submissionKeys = new Map(rows.filter(row => ordinal(row?.sequence) && row.sequence > 0
    && ordinal(row.generation) && finite(row.submittedAt))
    .map(row => [`${row.generation}:${row.sequence}`, row.neural]));
  const windowSamples = intervalValid ? samples.filter(sample => inWindow(sample?.submittedAt, startAt, stopAt)) : [];
  for (const sample of windowSamples) {
    const key = `${sample.generation}:${sample.sequence}`, state = latestAt(timedStates, sample.submittedAt, 'at');
    if (!finite(sample.ms) || sample.ms < 0 || !finite(sample.resolvedAt) || sample.resolvedAt < sample.submittedAt
      || !submissionKeys.has(key) || submissionKeys.get(key) !== sample.neural || seenSamples.has(key)
      || (state && state.generation > sample.generation)
      || (requireNeural && sample.neural !== true)) continue;
    seenSamples.add(key); validSamples.push(sample);
  }
  const windowMetrics = (name, from, to) => {
    const span = intervalValid ? Math.max(0, to - from) : null;
    const local = intervalValid ? observations.filter(row => inWindow(row.submittedAt, from, to)) : [];
    const submitted = local.filter(row => row.usefulSubmission).length;
    const rendered = intervalValid ? observations.filter(row => row.rendered
      && finite(row.opportunityAt) && row.opportunityAt >= from && (row.opportunityAt < to || (to === stopAt && row.opportunityAt === to))).length : 0;
    return { name, startAt: intervalValid ? from : null, stopAt: intervalValid ? to : null, durationMs: span,
      observations: local.length, usefulSubmissions: submitted, renderedOpportunities: rendered,
      submittedFps: span > 0 ? submitted * 1000 / span : null,
      renderedFps: span > 0 ? rendered * 1000 / span : null,
      sourceCallbackAgeMs: distribution(local.map(row => row.sourceCallbackAgeMs)),
      encodeSubmissionMs: distribution(local.map(row => row.encodeSubmissionMs)),
      opportunityDelayMs: distribution(local.map(row => row.opportunityDelayMs)),
      elapsedToOpportunityMs: distribution(local.map(row => row.elapsedToOpportunityMs)),
      opportunityVisualAgeMs: distribution(local.map(row => row.opportunityVisualAgeMs)),
      callbackLatencyMs: distribution(local.map(row => row.callbackLatencyMs)),
      gpuMs: distribution(validSamples.filter(sample => inWindow(sample.submittedAt, from, to)).map(sample => sample.ms)) };
  };
  const width = Math.min(durationMs ?? 0, 20000), midpoint = intervalValid ? (startAt + stopAt) / 2 : 0;
  const windows = { overall: windowMetrics('overall', startAt, stopAt),
    first20s: windowMetrics('first20s', startAt, startAt + width),
    middle20s: windowMetrics('middle20s', midpoint - width / 2, midpoint + width / 2),
    final20s: windowMetrics('final20s', stopAt - width, stopAt) };
  const fiveSecondWindows = [];
  for (let offset = 0; intervalValid && offset < durationMs; offset += 5000) {
    const end = Math.min(offset + 5000, durationMs);
    const ages = observations.filter(row => inWindow(row.submittedAt, startAt + offset, startAt + end))
      .map(row => row.opportunityVisualAgeMs);
    fiveSecondWindows.push({ startAt: startAt + offset, stopAt: startAt + end, complete: end - offset === 5000,
      age: distribution(ages) });
  }
  const growingWindows = [];
  for (let index = 2; index < fiveSecondWindows.length; index++) {
    const triple = fiveSecondWindows.slice(index - 2, index + 1);
    if (triple.every(window => window.complete && finite(window.age.p50))
      && triple[1].age.p50 - triple[0].age.p50 > period && triple[2].age.p50 - triple[1].age.p50 > period) {
      growingWindows.push(triple.map(window => window.startAt));
    }
  }
  const ageIncreaseMs = difference(windows.final20s.opportunityVisualAgeMs.p50, windows.first20s.opportunityVisualAgeMs.p50);
  const invalidFraction = counts.observations ? counts.invalidObservations / counts.observations : null;
  const stateIntervals = timedStates.slice(1).map((state, index) => state.at - timedStates[index].at);
  const stateCoverage = { count: selectedStates.length,
    leadingGapMs: timedStates.length ? timedStates[0].at - startAt : null,
    trailingGapMs: timedStates.length ? stopAt - timedStates.at(-1).at : null,
    intervalsMs: distribution(stateIntervals), maxGapMs: timedStates.length
      ? maxOrNull([timedStates[0].at - startAt, ...stateIntervals, stopAt - timedStates.at(-1).at]) : null };
  const badStates = selectedStates.filter(state => !ordinal(state.generation) || !RUNTIME_STATES.has(state.state)
    || !['neural', 'baseline'].includes(state.actualTier)
    || ['failed', 'unavailable'].includes(state.state) || (requireNeural
      && (state.actualTier !== 'neural' || ['fallback', 'probing', 'manual-baseline'].includes(state.state))));
  const snapshot = record?.snapshot;
  const errors = Array.isArray(record?.errors) ? [...record.errors] : ['Missing errors collection'];
  for (const error of [snapshot?.error, snapshot?.observerError, snapshot?.video?.error]) if (error != null) errors.push(error);
  if (Array.isArray(snapshot?.cleanupErrors)) errors.push(...snapshot.cleanupErrors);
  if (snapshot?.enhancementFailed === true || ['unavailable', 'error', 'failed'].includes(snapshot?.state)
    || ['failed', 'unavailable'].includes(snapshot?.runtime?.controller?.state)) errors.push('Enhancement/session failure in final snapshot');
  const criteria = {};
  const gate = (name, pass, observed, limit, reason) => { criteria[name] = { pass: Boolean(pass), observed, limit, reason }; };
  gate('duration', intervalValid && durationMs >= minDurationMs, durationMs, { minimumMs: minDurationMs }, 'Actual stopAt - startAt must cover the required observation duration');
  gate('rowCoverage', Array.isArray(record?.rows) && selected.length > 0 && counts.unboundedRows === 0, counts.unboundedRows,
    { maximumUnboundedRows: 0 }, 'Rows need finite submission timestamps to establish interval membership');
  gate('clocks', counts.clockErrors === 0 && selected.length > 0, counts.clockErrors, { maximum: 0 }, 'Same-task brackets and ordered callback/submission/media clocks must be valid');
  gate('identity', selected.length > 0 && counts.missingIdentities === 0 && counts.missingMetadata === 0,
    { missingIdentities: counts.missingIdentities, missingMetadata: counts.missingMetadata }, { maximum: 0 }, 'Every observed texture needs a recognized identity and finite native mediaTime');
  gate('sourceAge', windows.overall.sourceCallbackAgeMs.count > 0 && counts.misalignedIdentities === 0, counts.misalignedIdentities,
    { maximumAbsoluteMs: period, quantizationEpsilonMs: FRAME_QUANTIZATION_MS }, 'Known texture PTS must be within one source frame of native mediaTime in either direction');
  gate('freshness', counts.stalePresentations === 0 && counts.staleGenerations === 0 && counts.sequenceRegressions === 0,
    { stalePresentations: counts.stalePresentations, staleGenerations: counts.staleGenerations, sequenceRegressions: counts.sequenceRegressions },
    { maximum: 0 }, 'No old sequence/generation may be accepted as current visible output');
  gate('invalidFraction', finite(invalidFraction) && invalidFraction <= 0.01, invalidFraction, { maximum: 0.01 }, 'At most 1% of in-window observations may lack a valid rendered opportunity');
  gate('encodeSubmission', finite(windows.overall.encodeSubmissionMs.p95) && windows.overall.encodeSubmissionMs.p95 <= period,
    windows.overall.encodeSubmissionMs.p95, { maximumP95Ms: period }, 'Same-task software encode/submission p95 must fit one source frame');
  gate('opportunityP95', finite(windows.overall.opportunityDelayMs.p95) && windows.overall.opportunityDelayMs.p95 <= 2 * period,
    windows.overall.opportunityDelayMs.p95, { maximumP95Ms: 2 * period }, 'Next-rAF delay p95 must fit two source frames');
  gate('opportunityMax', finite(windows.overall.opportunityDelayMs.max) && windows.overall.opportunityDelayMs.max <= 250,
    windows.overall.opportunityDelayMs.max, { maximumMs: 250 }, 'Maximum observed in-window next-rAF delay must not exceed 250ms');
  gate('ageIncrease', finite(ageIncreaseMs) && ageIncreaseMs <= period, ageIncreaseMs, { maximumIncreaseMs: period }, 'Final20s minus first20s median visual age; decreases are not failures');
  gate('ageGrowth', fiveSecondWindows.length >= 3 && growingWindows.length === 0, growingWindows,
    { consecutiveWindows: 3, increases: 2, eachGreaterThanMs: period }, 'Three consecutive complete 5s windows with both median increases exceeding one frame fail');
  for (const name of ['overall', 'final20s']) for (const metric of ['submittedFps', 'renderedFps']) {
    gate(`${name}.${metric}`, finite(windows[name][metric]) && windows[name][metric] >= minimumFps,
      windows[name][metric], { minimumFps }, `${name} ${metric} uses actual event timestamps and the full window duration`);
  }
  gate('controllerCoverage', selectedStates.length > 0 && states.every(state => finite(state?.at))
    && selectedStates.every((state, index) => index === 0 || (state.at >= selectedStates[index - 1].at
      && state.generation >= selectedStates[index - 1].generation)), stateCoverage,
    { minimumSamples: 1 }, 'Controller observations must exist with ordered timestamps; sampling gaps are reported, not threshold-gated');
  gate('controller', badStates.length === 0 && (!requireNeural || counts.nonNeuralSubmissions === 0),
    { badStates, nonNeuralSubmissions: counts.nonNeuralSubmissions }, { requireNeural }, 'Neural qualification forbids baseline, fallback, probing or failure after warmup');
  gate('errors', errors.length === 0, errors, { maximum: 0 }, 'GPU, session, observer and cleanup errors fail the run');
  gate('gpuEvidence', validSamples.length > 0, validSamples.length, { minimumSamples: 1 }, 'Timestamp GPU evidence is required; CPU or opportunity time is not a substitute');
  const failed = Object.entries(criteria).filter(([, criterion]) => !criterion.pass).map(([name]) => name);
  return { outcome: failed.length ? 'FAIL' : 'PASS', reason: failed.length ? `Failed: ${failed.join(', ')}` : 'All software playback criteria passed', criteria,
    metrics: { fps, requireNeural, sourceFrameMs: period, minimumFps, startAt: finite(startAt) ? startAt : null,
      stopAt: finite(stopAt) ? stopAt : null, durationMs, reportedDurationMs: finite(record?.durationMs) ? record.durationMs : null,
      counts, invalidFraction, windows, observations, fiveSecondWindows, ageIncreaseMs, growingWindows, stateCoverage,
      states: selectedStates.map(state => ({ ...state })), gpuSamples: { inWindow: windowSamples.length, accepted: validSamples.length,
        rejected: windowSamples.length - validSamples.length, unbounded: samples.filter(sample => !finite(sample?.submittedAt)).length },
      callbackIntervalsMs: distribution(observations.map(row => row.callbackIntervalMs)), callbackGaps: null, decodeDropped: null,
      snapshotCounters: { decoderDrops: finite(snapshot?.runtime?.session?.decoderDrops) ? snapshot.runtime.session.decoderDrops : null,
        framesSkipped: finite(snapshot?.runtime?.session?.framesSkipped) ? snapshot.runtime.session.framesSkipped : null },
      resources: snapshot?.resources ? structuredClone(snapshot.resources) : null,
      scope: { claim: 'Software observations only; not physical scanout, pixels displayed, or audio synchronization',
        window: 'Submission cohort [startAt, stopAt); start is a perf-now boundary, not a callback anchor. Count the opening event; never subtract one. Duration is stopAt-startAt, not callback span or requested time.',
        rendered: 'Unique useful submissions with a latest, same-generation, visible, non-superseded next-rAF opportunity at or before stopAt. Endpoint state/submission events participate in freshness, not counts. Window rendered FPS is assigned by opportunityAt; submissions by submittedAt.',
        timing: 'Timing distributions use submission cohorts, including finite invalid observations. sourceCallbackAgeMs=(mediaTime-sourceIdentity/fps)*1000; encodeSubmissionMs=submittedAt-before; opportunityDelayMs=opportunityAt-submittedAt; elapsedToOpportunityMs=opportunityAt-before; opportunityVisualAgeMs=sourceCallbackAgeMs+elapsedToOpportunityMs. Out-of-window opportunities are censored, raw delays retained.',
        statistics: 'Finite values only, linear-interpolated percentiles; absent values remain null, observed counts may be zero. values preserves raw signed distributions without baseline subtraction.',
        gpu: 'Raw GPU timestamp ms, joined by sequence/generation, selected by sample.submittedAt (pre-encode CPU provenance, not row post-submit time). Resolutions after stop are allowed. No CPU-derived GPU durations.',
        gaps: 'Sequence and media/identity gaps span adjacent recorded successful submissions only, not all rVFC callbacks or decoder drops. No opening/terminal gaps are imputed; callbackGaps and interval decodeDropped are unmeasured.',
        snapshot: 'Final runtime counters are since session/reset, not observation-window deltas. Final pause may suspend the controller. Resource counts describe the final snapshot only.',
        controls: `${requireNeural ? 'Instrumented neural pipeline' : 'Instrumented pipeline control'}; does not infer raw/native-video-only rendering or texture bit-repeat counts` } } };
}