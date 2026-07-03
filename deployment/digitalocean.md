# DigitalOcean Deployment

This app runs well on a plain Ubuntu Droplet. Use Ubuntu 22.04 LTS so the pinned
Python packages install against Python 3.10 cleanly.

## 1. Create An SSH Key From Windows

Run this in PowerShell:

```powershell
ssh-keygen -t ed25519 -C "trading-agent-do" -f "$env:USERPROFILE\.ssh\do_trading_agent"
Get-Content "$env:USERPROFILE\.ssh\do_trading_agent.pub"
```

Copy the public key output and add it in DigitalOcean under Settings > Security >
SSH Keys.

## 2. Create The Droplet

Recommended settings:

- Image: Ubuntu 22.04 LTS x64
- Size: Basic shared CPU, 2 GB RAM minimum
- Region: closest to you, for example SFO3 for US West
- Authentication: SSH key
- Hostname: trading-agent
- Monitoring: enabled
- Backups: optional but recommended once the app is working
- Tags: trading-agent

Do not expose port 5001 publicly. The dashboard should stay bound to
127.0.0.1 and be accessed through an SSH tunnel.

## 3. Add A Cloud Firewall

Create a DigitalOcean Cloud Firewall and apply it to the `trading-agent` tag.

Inbound rules:

- SSH, TCP 22, source: your current public IP only

Outbound rules:

- Allow all outbound traffic

## 4. Bootstrap The Droplet

SSH in as root:

```powershell
ssh -i "$env:USERPROFILE\.ssh\do_trading_agent" root@<DROPLET_IP>
```

Then run:

```bash
adduser trader
usermod -aG sudo trader
rsync --archive --chown=trader:trader ~/.ssh /home/trader

apt update
apt upgrade -y
apt install -y git python3-venv python3-pip build-essential sqlite3 curl
timedatectl set-timezone America/Los_Angeles
```

Log out, then reconnect as the app user:

```powershell
ssh -i "$env:USERPROFILE\.ssh\do_trading_agent" trader@<DROPLET_IP>
```

## 5. Install The App

Clone the private repo:

```bash
git clone <YOUR_PRIVATE_REPO_URL> ~/xbox_skill
cd ~/xbox_skill
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

From your local machine, copy runtime-only files:

```powershell
scp -i "$env:USERPROFILE\.ssh\do_trading_agent" .env trading_data.db trader@<DROPLET_IP>:/home/trader/xbox_skill/
```

Keep `.env` and `trading_data.db` out of git.

## 6. Start The Services

On the Droplet:

```bash
cd ~/xbox_skill
sudo cp deployment/trading-dashboard.service.example /etc/systemd/system/trading-dashboard.service
sudo cp deployment/trading-digest.service.example /etc/systemd/system/trading-digest.service
sudo systemctl daemon-reload
sudo systemctl enable --now trading-dashboard trading-digest
```

Check health:

```bash
systemctl status trading-dashboard
systemctl status trading-digest
journalctl -u trading-dashboard -n 80 --no-pager
journalctl -u trading-digest -n 80 --no-pager
```

## 7. Open The Dashboard

From Windows, open an SSH tunnel:

```powershell
ssh -i "$env:USERPROFILE\.ssh\do_trading_agent" -L 5001:127.0.0.1:5001 trader@<DROPLET_IP>
```

Then open:

```text
http://127.0.0.1:5001
```

## 8. Useful Operations

Restart services:

```bash
sudo systemctl restart trading-dashboard trading-digest
```

Watch logs:

```bash
journalctl -u trading-dashboard -f
journalctl -u trading-digest -f
```

Back up the SQLite database from Windows:

```powershell
scp -i "$env:USERPROFILE\.ssh\do_trading_agent" trader@<DROPLET_IP>:/home/trader/xbox_skill/trading_data.db .\trading_data.backup.db
```

OpenClaw/WhatsApp should be enabled only after the dashboard and scheduler are
stable on the Droplet.
