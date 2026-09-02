#!/usr/bin/env node
/**
 * Emits a minimal single-Conv ONNX model, with no Python or onnx package.
 *
 * Why hand-rolled: the machine has neither `onnx` nor `torch`, and the model
 * needed for the Milestone 2 runtime comparison is one 3x3 convolution. ONNX
 * is protobuf, and a graph this small is a few nested length-delimited
 * messages, so emitting the bytes directly is less work — and far less
 * dependency surface — than installing a toolchain.
 *
 * The model is deliberately trivial and is a benchmark fixture, not a
 * super-resolution network:
 *
 *     input [1, C, H, W] float32
 *       -> Conv 3x3, pad 1, C -> C, with bias
 *       -> Relu
 *       -> output [1, C, H, W]
 *
 * Usage:
 *   node tools/make-onnx-conv.mjs <channels> <out.onnx>
 */

import { writeFileSync } from 'node:fs';

// ---------- protobuf primitives ----------

function varint(value) {
  const bytes = [];
  let v = BigInt(value);
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v > 0n) byte |= 0x80;
    bytes.push(byte);
  } while (v > 0n);
  return Buffer.from(bytes);
}

const tag = (field, wire) => varint((field << 3) | wire);
const fVarint = (field, value) => Buffer.concat([tag(field, 0), varint(value)]);
const fBytes = (field, buf) => Buffer.concat([tag(field, 2), varint(buf.length), buf]);
const fString = (field, str) => fBytes(field, Buffer.from(str, 'utf8'));

/** Packed repeated int64 (wire type 2). */
const fPackedInts = (field, values) =>
  fBytes(field, Buffer.concat(values.map((v) => varint(v))));

// ---------- ONNX messages ----------

const ELEM_FLOAT = 1;

/** AttributeProto with repeated ints. INTS = 7. */
function attrInts(name, ints) {
  return Buffer.concat([fString(1, name), fVarint(20, 7), fPackedInts(8, ints)]);
}

/** TensorShapeProto from concrete dimensions. */
function shape(dims) {
  return Buffer.concat(dims.map((d) => fBytes(1, fVarint(1, d))));
}

/** ValueInfoProto for a float tensor of the given shape. */
function valueInfo(name, dims) {
  const tensorType = Buffer.concat([fVarint(1, ELEM_FLOAT), fBytes(2, shape(dims))]);
  const typeProto = fBytes(1, tensorType);
  return Buffer.concat([fString(1, name), fBytes(2, typeProto)]);
}

/** TensorProto initializer holding float32 raw data. */
function initializer(name, dims, values) {
  const raw = Buffer.alloc(values.length * 4);
  values.forEach((v, i) => raw.writeFloatLE(v, i * 4));
  return Buffer.concat([
    ...dims.map((d) => fVarint(1, d)),
    fVarint(2, ELEM_FLOAT),
    fString(8, name),
    fBytes(9, raw),
  ]);
}

function node(inputs, outputs, opType, name, attributes = []) {
  return Buffer.concat([
    ...inputs.map((i) => fString(1, i)),
    ...outputs.map((o) => fString(2, o)),
    fString(3, name),
    fString(4, opType),
    ...attributes.map((a) => fBytes(5, a)),
  ]);
}

function build(channels, height, width) {
  const count = channels * channels * 9;
  // Deterministic small weights: a real network's values, not a delta kernel,
  // so nothing can be constant-folded away.
  const weights = Array.from({ length: count }, (_, i) => (Math.sin(i * 0.37) * 0.08));
  const biases = Array.from({ length: channels }, (_, i) => (i % 5) * 0.01);

  const conv = node(
    ['input', 'W', 'B'],
    ['conv_out'],
    'Conv',
    'conv0',
    [
      attrInts('dilations', [1, 1]),
      attrInts('kernel_shape', [3, 3]),
      attrInts('pads', [1, 1, 1, 1]),
      attrInts('strides', [1, 1]),
      Buffer.concat([fString(1, 'group'), fVarint(20, 2), fVarint(3, 1)]), // INT = 2
    ],
  );
  const relu = node(['conv_out'], ['output'], 'Relu', 'relu0');

  const graph = Buffer.concat([
    fBytes(1, conv),
    fBytes(1, relu),
    fString(2, 'aethervsr_conv_bench'),
    fBytes(5, initializer('W', [channels, channels, 3, 3], weights)),
    fBytes(5, initializer('B', [channels], biases)),
    fBytes(11, valueInfo('input', [1, channels, height, width])),
    fBytes(12, valueInfo('output', [1, channels, height, width])),
  ]);

  return Buffer.concat([
    fVarint(1, 7), // ir_version 7
    fString(2, 'aethervsr'),
    fBytes(7, graph),
    fBytes(8, Buffer.concat([fString(1, ''), fVarint(2, 13)])), // opset 13
  ]);
}

const channels = Number(process.argv[2] ?? 16);
const out = process.argv[3] ?? `conv${channels}.onnx`;
const height = Number(process.argv[4] ?? 720);
const width = Number(process.argv[5] ?? 1280);

const bytes = build(channels, height, width);
writeFileSync(out, bytes);
console.log(
  `wrote ${out}: ${bytes.length} bytes — Conv ${channels}->${channels} 3x3 pad1 + Relu, input [1,${channels},${height},${width}]`,
);
