#!/bin/bash
# Start Cloudflare Tunnel for LingBot-World-V2 Web UI (Port 7860)

BIN="/home/sims/bin/cloudflared"
if [ ! -f "$BIN" ]; then
    mkdir -p /home/sims/bin
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /home/sims/bin/cloudflared
    chmod +x /home/sims/bin/cloudflared
fi

echo "🚀 Starting Cloudflare Edge Tunnel on http://127.0.0.1:7860 ..."
exec /home/sims/bin/cloudflared tunnel --url http://127.0.0.1:7860
