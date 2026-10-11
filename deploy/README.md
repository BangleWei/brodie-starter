# Running Chef Brodie always-on on a cloud server

Two processes run on one small Linux server, so the FaceTime line answers even when the Mac is off:
- `chef-worker`: the Brodie worker (Gemini Live voice and vision), registered with LiveKit as agent `chef`.
- `chef-line`: `chert dev --agent chef`, which holds the Chert FaceTime line and routes each call to the worker.

Both are systemd services with `Restart=always`, so they come back after a crash, a network drop or a reboot.

## 1. Server
Ubuntu 24.04, 2 vCPU / 4 GB RAM, in a US East region (the LiveKit project is in US East). Any provider works, for example
DigitalOcean, Hetzner or Vultr. Add your SSH key when creating it.

```bash
ssh root@SERVER
adduser --disabled-password --gecos "" chef && mkdir -p /home/chef/.ssh && cp ~/.ssh/authorized_keys /home/chef/.ssh/ && chown -R chef:chef /home/chef/.ssh
apt-get update && apt-get install -y git curl build-essential libgl1 libglib2.0-0
```

## 2. Code and private files
```bash
ssh chef@SERVER 'git clone https://github.com/BangleWei/brodie-starter.git ~/brodie-rt-coach'
# From the Mac: licensed files and secrets go over scp, never through git.
scp -r ~/brodie-rt-coach/data ~/brodie-rt-coach/models chef@SERVER:~/brodie-rt-coach/
scp ~/brodie-rt-coach/.env.local chef@SERVER:~/brodie-rt-coach/.env.local
ssh chef@SERVER 'chmod 600 ~/brodie-rt-coach/.env.local'
```
`.env.local` on the server needs `AGENT_NAME=chef`. For your own Google Gemini key, set `GEMINI_API_KEY` to it and delete the
`GEMINI_BASE_URL` line (no base URL means the worker talks to Google directly). To keep using the Brodie relay, leave both as they are.

## 3. Install and sign in
```bash
ssh chef@SERVER
~/brodie-rt-coach/deploy/setup.sh
. ~/.nvm/nvm.sh && cd ~/brodie-rt-coach && chert login --device   # approve the code at https://facetime.trychert.com/cli
chert doctor --agent chef                                          # pick the dedicated line's project if asked
```
The Chert sign-in lasts 30 days. Repeat `chert login --device` monthly (set a reminder), then `sudo systemctl restart chef-line`.

## 4. Turn it on
```bash
sudo cp ~/brodie-rt-coach/deploy/chef-*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now chef-worker chef-line
journalctl -u chef-worker -u chef-line -f      # watch for "registered worker" and "in a waiting room"
```
**Stop the Mac's worker and `chert dev` first**, so the server is the only agent named `chef`.

## Day to day
- Update: `cd ~/brodie-rt-coach && git pull && sudo systemctl restart chef-worker chef-line` (no hot reload).
- Logs: `journalctl -u chef-worker -f` (calls, what Brodie said), `journalctl -u chef-line -f` (incoming calls).
- Pause: `sudo systemctl stop chef-line` frees the line; `start` puts it back.

## Test
With the Mac off, FaceTime Video the dedicated number: Brodie greets you. Then `sudo reboot`, wait a minute, and call again.
