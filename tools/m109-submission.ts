import type { VideoPipeline } from '../src/core/pipeline.js';

declare const __AETHERVSR_TEST__: boolean;

export interface SubmissionIdentity {
  owner: string;
  sourceGeneration: number;
  geometryGeneration: number;
  backingWidth: number;
  backingHeight: number;
  authorized: boolean;
}

export interface SuccessfulFrameSubmission extends SubmissionIdentity {
  kind: 'successfulFrameSubmission';
  boundary: 'queue.submit returned; not GPU completion or scanout';
  sequence: number;
  frameGeneration: number;
  mediaTime: number;
  sourceWidth: number;
  sourceHeight: number;
  validForRecovery: boolean;
}

export function observeSuccessfulSubmissions(
  pipeline: VideoPipeline,
  readIdentity: () => SubmissionIdentity | null,
  receive: (record: SuccessfulFrameSubmission) => void,
): () => void {
  const sequenceState = pipeline as unknown as { readonly submissionSequence: number };
  if (!Number.isSafeInteger(sequenceState.submissionSequence)) throw new Error('Submission sequence unavailable');
  const previous = pipeline.onFrame;
  let lastSequence = sequenceState.submissionSequence;
  let active = true;
  const observer: NonNullable<VideoPipeline['onFrame']> = tick => {
    const sequence = sequenceState.submissionSequence;
    const fresh = Number.isSafeInteger(sequence) && sequence > lastSequence;
    if (fresh) lastSequence = sequence;
    const frameGeneration = pipeline.timingGeneration;
    const identity = active ? readIdentity() : null;
    const before = identity === null ? null : { ...identity };
    previous?.(tick);
    if (!active || pipeline.error || !pipeline.running || before === null ||
      !fresh || sequence !== sequenceState.submissionSequence) return;
    const after = readIdentity();
    if (after === null) return;
    receive({ ...before, kind: 'successfulFrameSubmission',
      boundary: 'queue.submit returned; not GPU completion or scanout',
      sequence, frameGeneration, mediaTime: tick.mediaTime,
      sourceWidth: tick.size.width, sourceHeight: tick.size.height,
      validForRecovery: before.authorized && after.authorized &&
        before.owner === after.owner && before.sourceGeneration === after.sourceGeneration &&
        before.geometryGeneration === after.geometryGeneration && frameGeneration === pipeline.timingGeneration &&
        before.backingWidth === after.backingWidth && before.backingHeight === after.backingHeight &&
        before.backingWidth === tick.size.width * 2 && before.backingHeight === tick.size.height * 2 });
  };
  pipeline.onFrame = observer;
  return () => {
    active = false;
    if (pipeline.onFrame === observer) pipeline.onFrame = previous;
  };
}

if (typeof __AETHERVSR_TEST__ !== 'undefined' && __AETHERVSR_TEST__ &&
  !Object.hasOwn(globalThis, '__AETHERVSR_SUCCESSFUL_SUBMISSION__')) {
  Object.defineProperty(globalThis, '__AETHERVSR_SUCCESSFUL_SUBMISSION__', { value: observeSuccessfulSubmissions });
}