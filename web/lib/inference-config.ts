export const BROWSER_MODEL_DTYPE = 'q4' as const;

// Override sampling defaults shipped with the model. The metrics describe the
// raw distribution that greedily selected each token, so no processor may alter
// that selection after the fact.
export const DETERMINISTIC_GENERATION = Object.freeze({
  do_sample: false,
  repetition_penalty: 1.0,
  temperature: 1.0,
  top_k: 0,
  top_p: 1.0,
});
