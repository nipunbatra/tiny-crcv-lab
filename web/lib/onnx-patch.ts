const HIDDEN_NAME = '/model/norm/Mul_1_output_0';

function readVarint(bytes: Uint8Array, offset: number): { value: number; next: number } {
  let value = 0;
  let multiplier = 1;
  let cursor = offset;
  while (cursor < bytes.length) {
    const byte = bytes[cursor++];
    value += (byte & 0x7f) * multiplier;
    if ((byte & 0x80) === 0) return { value, next: cursor };
    multiplier *= 128;
    if (multiplier > Number.MAX_SAFE_INTEGER) throw new Error('ONNX varint exceeds JavaScript safe integer range');
  }
  throw new Error('Truncated ONNX varint');
}

function varint(value: number): Uint8Array {
  const output: number[] = [];
  let remaining = value;
  do {
    let byte = remaining % 128;
    remaining = Math.floor(remaining / 128);
    if (remaining > 0) byte |= 0x80;
    output.push(byte);
  } while (remaining > 0);
  return Uint8Array.from(output);
}

function bytes(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.length; }
  return result;
}

function field(number: number, wireType: number, payload: Uint8Array): Uint8Array {
  const key = varint(number * 8 + wireType);
  return wireType === 2 ? bytes(key, varint(payload.length), payload) : bytes(key, payload);
}

function textField(number: number, value: string): Uint8Array {
  return field(number, 2, new TextEncoder().encode(value));
}

function messageField(number: number, ...content: Uint8Array[]): Uint8Array {
  return field(number, 2, bytes(...content));
}

function hiddenOutputValueInfo(elementType: 1 | 10): Uint8Array {
  const dimensionParam = (name: string) => messageField(1, textField(2, name));
  const dimensionValue = (value: number) => messageField(1, field(1, 0, varint(value)));
  const shape = messageField(2, dimensionParam('batch_size'), dimensionParam('sequence_length'), dimensionValue(896));
  const tensorType = messageField(1, field(1, 0, varint(elementType)), shape);
  const valueInfo = bytes(textField(1, HIDDEN_NAME), messageField(2, tensorType));
  return messageField(12, valueInfo); // GraphProto.output = field 12
}

function locateGraphHeader(head: Uint8Array): { lengthStart: number; graphStart: number; graphLength: number } {
  let cursor = 0;
  while (cursor < head.length) {
    const key = readVarint(head, cursor);
    cursor = key.next;
    const fieldNumber = Math.floor(key.value / 8);
    const wireType = key.value % 8;
    if (fieldNumber === 7 && wireType === 2) {
      const lengthStart = cursor;
      const length = readVarint(head, cursor);
      return { lengthStart, graphStart: length.next, graphLength: length.value };
    }
    if (wireType === 0) cursor = readVarint(head, cursor).next;
    else if (wireType === 1) cursor += 8;
    else if (wireType === 2) { const length = readVarint(head, cursor); cursor = length.next + length.value; }
    else if (wireType === 5) cursor += 4;
    else throw new Error(`Unsupported ONNX protobuf wire type ${wireType}`);
  }
  throw new Error('Could not locate ModelProto.graph in the ONNX header');
}

/**
 * Return an ONNX Blob with one extra graph output, without copying the weights.
 * q4 activations are float32 (element type 1); q4f16 activations are float16 (10).
 */
export async function exposeHiddenOutput(model: Blob, elementType: 1 | 10): Promise<Blob> {
  const head = new Uint8Array(await model.slice(0, Math.min(model.size, 1024 * 1024)).arrayBuffer());
  const { lengthStart, graphStart, graphLength } = locateGraphHeader(head);
  const graphEnd = graphStart + graphLength;
  if (graphEnd > model.size) throw new Error('ONNX graph length extends beyond the model file');
  const output = hiddenOutputValueInfo(elementType);
  const updatedLength = varint(graphLength + output.length).slice().buffer as ArrayBuffer;
  const outputBuffer = output.slice().buffer as ArrayBuffer;
  return new Blob([
    model.slice(0, lengthStart),
    updatedLength,
    model.slice(graphStart, graphEnd),
    outputBuffer,
    model.slice(graphEnd),
  ], { type: 'application/octet-stream' });
}

export { HIDDEN_NAME };
