# Model lock (team record)

This is your team's record of the model you serve for the rest of the course.

## The locked model

- Model id: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- Quantisation: `awq`
- Why this one: passed the smoke test cleanly (10/10, distractor clean), VRAM stayed within about 1GB of fp16 with the freed weight-memory going to extra KV-cache blocks, and tokens/s (39.7) was in the expected range for AWQ's fused kernels rather than the slowdown seen with day 1's bitsandbytes path.

## The launch flags

The exact vLLM flags your team runs. Copy them from the SERVER_ARGS you launched with.

--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype half --max-model-len 4096 \
--gpu-memory-utilization 0.85 \
--quantization awq \
--enable-auto-tool-choice --tool-call-parser hermes

- Tool-call parser: `hermes` (Qwen2.5, Hermes-3) or `llama3_json` (Llama-3.1)

## The smoke score

- Score (valid behaviours out of 10): `10`
- Distractor stayed call-free in the majority: `yes`
- Passed the gate (>= 8/10 and distractor majority clean): `yes`
- Measured against: `AWQ`

## Quality spot check note

- AWQ handled the inference-server summary, the sentence refactor, and the rollback-steps prompts reasonably well, but showed two real fidelity issues across the five-prompt check: it hallucinated "Cairo" instead of Riyadh on the tool-call prompt and rambled into a generic API tutorial rather than naming two clear tool calls, and its plain-language quantisation explanation for a non-technical manager came out garbled and not usable as written. The refactor answer also lost the "busy but not productive" distinction from the original sentence, collapsing it into "overloaded with work." Net judgment: solid enough to lock given the strong smoke score, but not flawless on prompts needing precise wording or grounded specifics.