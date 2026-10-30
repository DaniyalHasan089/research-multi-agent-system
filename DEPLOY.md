# Server Deployment Guide

> **Assumes**: Ubuntu 22.04/24.04 server, connected via SSH, deploying to `/root/`.

---

## Step 1 — Install Docker

```bash
apt update && apt upgrade -y

curl -fsSL https://get.docker.com | sh

docker --version        # confirm install
docker compose version  # confirm plugin
```

---

## Step 2 — Clone the Repository

```bash
cd /root
git clone https://github.com/YOUR_USERNAME/research-multi-agent-system.git
cd research-multi-agent-system
```

---

## Step 3 — Create Your `.env` File

```bash
cp .env.example .env
nano .env
```

Fill in your keys, then save (`Ctrl+O` → Enter → `Ctrl+X`):

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_real_key_here
TAVILY_API_KEY=your_real_key_here
TOP_K_PAPERS=15
```

---

## Step 4 — Build & Start

```bash
docker compose up --build -d
```

First build takes ~3–5 minutes. Watch progress:

```bash
docker compose logs -f
```

Wait until you see `Application startup complete` in the api logs, then press `Ctrl+C`.

---

## Step 5 — Verify It's Running

```bash
docker compose ps
```

Both services should show **healthy**:

```
NAME                     STATUS
research-agent-api       Up (healthy)
research-agent-nginx     Up (healthy)
```

Open in browser: **`http://YOUR_SERVER_IP`**

---

## Step 6 — Keep It Running After Reboot

Docker's `restart: unless-stopped` policy handles this automatically.  
To confirm it survives a reboot:

```bash
reboot
# reconnect via SSH after ~30 s
docker compose -f /root/research-multi-agent-system/compose.yaml ps
```

---

## Common Commands

| Task | Command |
|------|---------|
| Stop app | `docker compose down` |
| Start app | `docker compose up -d` |
| Live logs | `docker compose logs -f` |
| Rebuild after code change | `git pull && docker compose up --build -d` |
| Check disk usage | `docker system df` |
| Free unused images | `docker system prune -f` |

---

## Firewall (if enabled)

Allow HTTP traffic on port 80:

```bash
ufw allow 80/tcp
ufw allow 22/tcp   # keep SSH open!
ufw enable
ufw status
```

---

> **HTTPS / domain**: Point your domain's A record to the server IP, then add Certbot + Nginx SSL — see the production notes in `DOCKER_README.md`.

