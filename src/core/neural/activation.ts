/**
 * Pointwise nonlinearities the convolution kernels can fuse into their store.
 *
 * Fused rather than applied by a separate pass: an activation reads and writes
 * a whole activation tensor, which at 1280x720 C16 is 29.5 MB each way, and the
 * arithmetic is free by comparison. Milestone 3 measured `tanh` on 14.7M
 * outputs at 3.2% of a convolution.
 */
export type Activation = 'none' | 'relu' | 'tanh';
