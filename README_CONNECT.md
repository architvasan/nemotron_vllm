# Nemotron-3.5-Super-VL-120B on JLSE — connection guide

## What's running
- **Model**: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` (hybrid Mamba-2 + Transformer + MoE, vision+text)
- **Engine**: vLLM 0.32.0 (apptainer: `vllm-0.32.0.sif`), tensor-parallel 4
- **Node**: `hopper00` (gpu_h100 queue), 4× H100 80GB
- **API**: OpenAI-compatible, `http://hopper00:8000/v1`
- **Served model name**: `nemotron-3.5-super-vl`
- **Context**: 32768 tokens; reasoning model (returns a `reasoning` field + `content`)

## Tunnel from your laptop
The vLLM server listens on the **compute node** `hopper00`, reachable from the login node.
Forward a local port straight through the login node to the compute node:

    ssh -N -L 8000:hopper00:8000 avasan@jlselogin6.ftm.alcf.anl.gov

If you reach JLSE through a bastion/jump host, use ProxyJump:

    ssh -N -L 8000:hopper00:8000 -J <bastion> avasan@jlselogin6.ftm.alcf.anl.gov

Leave that running. Then in hermes (or any OpenAI client) point at:
- **base_url**: `http://localhost:8000/v1`
- **model**:    `nemotron-3.5-super-vl`
- **api_key**:  any non-empty string (vLLM ignores it), e.g. `EMPTY`

## Quick test from the laptop (after the tunnel is up)
    curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model":"nemotron-3.5-super-vl",
      "messages":[{"role":"user","content":"What is 17*24? Reply with just the number."}],
      "max_tokens":1024, "temperature":0
    }'

Note: it's a reasoning model — give it a generous `max_tokens` (>=512) or it will
spend the whole budget "thinking" and return empty `content`. The thinking text is in
`choices[0].message.reasoning`; the final answer is in `choices[0].message.content`.

## Vision (it's a VL model)
    {"role":"user","content":[
      {"type":"text","text":"Describe this image."},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,...."}}
    ]}

## Relaunch after the 6h job ends (or if it dies)
    qsub -q gpu_h100 -t 360 -n 1 /vast/users/avasan/nemotron/serve_nemotron.sh
Then check which node it landed on (usually hopper00 again) and update your tunnel:
    cat /vast/users/avasan/nemotron/ENDPOINT.txt      # shows NODE/PORT
    qstat -u avasan                                   # shows Location column

## Files (all under /vast/users/avasan/nemotron)
- `model/`              — the 249 GB weights (63 safetensors shards + configs)
- `vllm-0.32.0.sif`     — the vLLM container (apptainer)
- `serve_nemotron.sh`   — the Cobalt serve job
- `ENDPOINT.txt`        — current node/port, written each launch
- `logs/`               — build/download logs
- stdout/stderr of the serve job: `~/<jobid>.output` / `~/<jobid>.error`

## Tuning knobs (edit serve_nemotron.sh)
- Longer context: raise `--max-model-len` (KV room is ~1.97M tokens total at 32k).
- More throughput: raise `--max-num-seqs`.
- Faster decode: add MTP speculative decoding (per model card):
  `--speculative_config.method mtp --speculative_config.num_speculative_tokens 3`
