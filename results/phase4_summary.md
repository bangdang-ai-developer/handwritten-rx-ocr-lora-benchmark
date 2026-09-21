# Phase 4 — PaddleOCR-VL: EXCLUDED FROM THE BENCHMARK (upstream compatibility bug)

Run on Kaggle (kernel v12, 16/09/2026). Result: `TypeError: create_causal_mask() got an unexpected keyword argument 'inputs_embeds'`, an error occurring inside the `modeling_paddleocr_vl.py` code itself (loaded via `trust_remote_code=True` from PaddlePaddle's HF repo, not this project's own code).

## Investigated — a known community bug with no fix yet

Official discussion on HuggingFace: ["Newest commit breaks compatibility with transformers==4.57.6, while 5.3.0 is broken as well"](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/discussions/22) (closed on 30/04, with no documented solution). Direct quote: *"Anything above `transformers==5` with `AutoModelForImageTextToText` has seemingly always been broken."* Another user reported "locking at `4.57.6`" as a temporary workaround, before a new commit to the model repo itself broke that workaround too.

**Conclusion:** this is not a bug caused by this project's T4/dtype/prompt setup (all lessons from Phase 3 were already applied: pinning `transformers==4.57.0`, using `float16`, debugging on 5 images before a full run) — rather, it is an internal API conflict between PaddleOCR-VL's custom code and the currently available `transformers` versions, with no version combination confirmed to work stably as of 16/09/2026.

## Decision

**Exclude PaddleOCR-VL from the zero-shot benchmark** — no further time will be invested debugging an unresolved upstream bug. The code that calls the model remains in `notebooks/kaggle_benchmark.py` (disabled via an early `return`, with an explanatory comment) so it can easily be retried if PaddlePaddle releases a fix before submission. Moving on to **Phase 5: Qwen2.5-VL-3B-Instruct** — a model included in mainline `transformers` (no `trust_remote_code` needed), with a much more stable ecosystem.

## Note for the paper

This remains a data point worth including in the Limitations/Discussion: not every "latest" VLM-OCR model is actually ready for immediate use in research practice — dependence on `trust_remote_code` and the fast pace of development of these repos can create real reproducibility barriers, an observation of value to the OCR/benchmarking community.
