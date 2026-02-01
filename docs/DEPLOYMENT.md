# VPS Deployment Guide

Complete guide for deploying the trading bot on a VPS with IC Markets.

## Recommended VPS Providers

| Provider | Location | Price | Latency to IC Markets |
|----------|----------|-------|----------------------|
| **Vultr** | Sydney | ~$6/mo | <1ms |
| **AWS Lightsail** | Sydney | ~$5/mo | <5ms |
| **Contabo** | Australia | ~$8/mo | <10ms |
| **ForexVPS** | Equinix NY4 | ~$25/mo | Ultra-low |

**Recommended**: Sydney-based VPS for best latency to IC Markets Australia servers.

---

## Option 1: MetaTrader 5 Setup (Recommended)

### Step 1: Set Up Windows VPS

IC Markets MT5 requires Windows. Get a Windows VPS:

```
Recommended Specs:
- Windows Server 2019/2022 or Windows 10/11
- 2 vCPU minimum
- 4 GB RAM minimum
- 50 GB SSD
```

### Step 2: Install MetaTrader 5

1. Download MT5 from IC Markets: https://www.icmarkets.com/au/trading-platforms/metatrader-5
2. Install MT5 on the VPS
3. Log into your IC Markets account in MT5
4. Enable Algo Trading: `Tools > Options > Expert Advisors > Allow Algo Trading`

### Step 3: Install Python & Bot

Open PowerShell on VPS:

```powershell
# Install Python (if not installed)
# Download from https://www.python.org/downloads/

# Clone/copy your bot to VPS
cd C:\TradingBot

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install MetaTrader5
```

### Step 4: Configure for Live Trading

Edit `config/settings.json`:

```json
{
  "dry_run": {
    "enabled": false
  },
  "broker": {
    "server": "ICMarketsSC-Live01"
  }
}
```

Create `config/credentials.json` (NEVER commit this!):

```json
{
  "mt5_login": 12345678,
  "mt5_password": "your_password",
  "mt5_server": "ICMarketsSC-Live01"
}
```

### Step 5: Create Startup Script

Create `run_live.py`:

```python
import asyncio
import json
from src.bot import TradingBot
from src.mt5_client import MT5Client
from src.api_client import APIClientWithRetry

async def main():
    # Load credentials
    with open('config/credentials.json') as f:
        creds = json.load(f)
    
    # Create MT5 client
    mt5_client = MT5Client()
    await mt5_client.connect(
        login=creds['mt5_login'],
        password=creds['mt5_password'],
        server=creds['mt5_server']
    )
    
    # Wrap with retry logic
    api_client = APIClientWithRetry(mt5_client)
    
    # Create bot with live client
    bot = TradingBot(dry_run=False)
    bot.api_client = api_client
    
    await bot.start()
    
    # Keep running
    while bot._is_running:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 6: Run as Windows Service

Install NSSM (Non-Sucking Service Manager):

```powershell
# Download from https://nssm.cc/download
# Extract to C:\nssm

# Install as service
C:\nssm\nssm.exe install TradingBot "C:\TradingBot\venv\Scripts\python.exe" "C:\TradingBot\run_live.py"

# Start service
C:\nssm\nssm.exe start TradingBot

# Check status
C:\nssm\nssm.exe status TradingBot
```

---

## Option 2: Linux VPS with cTrader API

If you prefer Linux or use cTrader:

### Step 1: Set Up Linux VPS

```bash
# Ubuntu 22.04 recommended
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3.10-venv python3-pip -y
```

### Step 2: Clone Bot

```bash
cd /opt
git clone <your-repo> tradingbot
cd tradingbot

python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: cTrader Open API Setup

1. Register at https://openapi.ctrader.com/
2. Create an application to get Client ID and Secret
3. Connect your IC Markets cTrader account

Create `config/ctrader_credentials.json`:

```json
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "access_token": "your_access_token",
  "account_id": "your_account_id"
}
```

### Step 4: Run with Systemd

Create `/etc/systemd/system/tradingbot.service`:

```ini
[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
User=tradingbot
WorkingDirectory=/opt/tradingbot
ExecStart=/opt/tradingbot/venv/bin/python run_live.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot
sudo systemctl status tradingbot
```

---

## Security Best Practices

### 1. Credential Management

```bash
# NEVER store passwords in code
# Use environment variables or encrypted config

# On Windows
[System.Environment]::SetEnvironmentVariable("MT5_PASSWORD", "xxx", "User")

# On Linux
echo 'export MT5_PASSWORD="xxx"' >> ~/.bashrc
```

### 2. Firewall Rules

```bash
# Linux - Only allow SSH
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw enable

# Windows - Use Windows Firewall
```

### 3. VPS Monitoring

Set up alerts for:
- VPS downtime
- High CPU/memory usage
- Unexpected bot stops

Recommended: Use UptimeRobot (free) or Datadog.

---

## Connecting to IC Markets

### MT5 Server Names

| Account Type | Server |
|-------------|--------|
| Demo | ICMarketsSC-Demo |
| Live (Standard) | ICMarketsSC-Live01, Live02, etc. |
| Live (Raw Spread) | ICMarketsEU-Live01, etc. |

### API Access

1. Log into IC Markets Client Portal
2. Go to Trading Accounts
3. Note your account number and server
4. Use existing MT5 password

---

## Monitoring on VPS

### View Logs

```powershell
# Windows
Get-Content C:\TradingBot\logs\trading_bot.log -Tail 50 -Wait

# Linux
tail -f /opt/tradingbot/logs/trading_bot.log
```

### Check Bot Status

```powershell
# Windows - Check service
Get-Service TradingBot

# Linux - Check systemd
systemctl status tradingbot
journalctl -u tradingbot -f
```

### Remote Dashboard

For remote monitoring, consider:

1. **Telegram Bot** - Get trade alerts on your phone
2. **Web Dashboard** - Build with Flask/FastAPI
3. **Email Alerts** - For critical events

---

## Troubleshooting

### MT5 Won't Connect

```
Error: "MT5 initialization failed"
```

Solutions:
1. Ensure MT5 terminal is running
2. Check "Allow Algo Trading" is enabled
3. Verify login credentials
4. Try running as Administrator

### Connection Drops

```
Error: "Connection lost"
```

Solutions:
1. Check VPS network stability
2. Enable MT5 auto-reconnect
3. Bot has automatic retry logic
4. Consider VPS closer to broker servers

### Order Rejected

```
Error: "Order rejected: No money"
```

Solutions:
1. Check account balance
2. Reduce lot size
3. Verify leverage settings
4. Check margin requirements

---

## Cost Estimation

| Item | Monthly Cost |
|------|-------------|
| VPS (Sydney) | $5-25 |
| IC Markets Account | $0 (no platform fees) |
| Total | ~$5-25/month |

---

## Quick Start Checklist

- [ ] Get VPS (Windows for MT5, Linux for cTrader)
- [ ] Install MT5 and log into IC Markets
- [ ] Copy bot files to VPS
- [ ] Install Python dependencies
- [ ] Configure credentials (secure!)
- [ ] Test in dry-run mode first
- [ ] Start with minimum lot size (0.01)
- [ ] Set up monitoring/alerts
- [ ] Run as background service
- [ ] Monitor logs regularly
