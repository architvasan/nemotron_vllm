#!/bin/bash
# 04 - Test the endpoint. Pass the base host:port as $1.
#   On the login node, hit the compute node directly:   ./04_test_inference.sh hopper00:8000
#   On your laptop (after the ssh -L tunnel),            ./04_test_inference.sh localhost:8000
set -euo pipefail
HP=${1:-localhost:8000}
URL="http://$HP"
MODEL=nemotron-3.5-super-vl

echo "=== 1) is it up? GET /v1/models ==="
curl -sS -m 15 "$URL/v1/models" | python3 -m json.tool

echo; echo "=== 2) chat completion (reasoning model: give it room, parse reasoning + content) ==="
curl -sS -m 120 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",
       \"messages\":[{\"role\":\"user\",\"content\":\"What is 17*24? Reply with just the number.\"}],
       \"max_tokens\":1024, \"temperature\":0}" \
| python3 -c "
import sys,json
d=json.load(sys.stdin); c=d['choices'][0]
print('finish_reason:', c['finish_reason'])
print('reasoning_tokens:', d['usage']['completion_tokens_details'].get('reasoning_tokens'))
print('--- reasoning ---'); print(c['message'].get('reasoning'))
print('--- content ---');   print(c['message'].get('content'))
"

echo; echo "=== 3) streaming check (first few SSE chunks) ==="
curl -sS -N -m 60 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"max_tokens\":256,\"temperature\":0,
       \"messages\":[{\"role\":\"user\",\"content\":\"Say hello in 3 languages.\"}]}" \
| head -8
