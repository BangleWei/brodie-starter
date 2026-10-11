#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu server, run as user "chef" from /home/chef/brodie-rt-coach.
# Does not touch secrets: .env.local, data/ and models/ are copied in separately (see deploy/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
if [ ! -d "$HOME/.nvm" ]; then curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash; fi
. "$HOME/.nvm/nvm.sh"
nvm install 24.21.0 && nvm alias default 24.21.0
npm i -g @trychert/facetime@0.1.0-beta.5
for f in .env.local data/cues_audio_gemini; do [ -e "$f" ] || echo "MISSING: $f (copy it from the Mac, see deploy/README.md)"; done
echo "Setup done. Next: chert login --device, then enable the services."
