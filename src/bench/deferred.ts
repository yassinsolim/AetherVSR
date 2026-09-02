/** A promise with its resolvers exposed, for linear async control flow. */
export interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
}

/**
 * Local stand-in for `Promise.withResolvers()`.
 *
 * The standard method would be preferable, but it requires Chrome 119+ while
 * the project's stated support floor is Chrome 113+ (the first release with
 * WebGPU). Raising `lib` to ES2024 to type it would silently authorise every
 * other post-floor API too, so the executor form is confined to this one
 * function and call sites stay flat.
 */
export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Resolves after `ms` milliseconds. */
export function delay(ms: number): Promise<void> {
  const { promise, resolve } = deferred<void>();
  setTimeout(resolve, ms);
  return promise;
}
