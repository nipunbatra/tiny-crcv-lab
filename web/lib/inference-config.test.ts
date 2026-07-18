import { describe, expect, it } from 'vitest';
import { BROWSER_MODEL_DTYPE, DETERMINISTIC_GENERATION } from './inference-config';

describe('browser inference contract', () => {
  it('uses the same graph precision for WebGPU and WASM', () => {
    expect(BROWSER_MODEL_DTYPE).toBe('q4');
  });

  it('neutralizes model-card sampling and repetition defaults', () => {
    expect(DETERMINISTIC_GENERATION).toEqual({
      do_sample: false,
      repetition_penalty: 1,
      temperature: 1,
      top_k: 0,
      top_p: 1,
    });
  });
});
