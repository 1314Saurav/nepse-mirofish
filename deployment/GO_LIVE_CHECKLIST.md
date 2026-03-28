# NEPSE MiroFish — Go-Live Checklist

> Complete every item below before deploying real capital.
> Run `make deploy-readiness` and `make go-live-check` to auto-fill where possible.

---

## Pre-Conditions (must ALL be checked before deploying real capital)

### 📊 Paper Trading Performance

- [ ] Minimum 20 trading days (4 weeks) completed
- [ ] Paper return is positive
- [ ] Paper alpha vs NEPSE index is positive
- [ ] Maximum drawdown < 15% during paper period
- [ ] No single week loss > 7%

### 🎯 Signal Quality

- [ ] 5-day signal accuracy ≥ 55%
- [ ] 10-day signal accuracy ≥ 50%
- [ ] MiroFish quality flags < 20% of signals
- [ ] Zep memory agent retention confirmed working
- [ ] At least 3 full BULL/BEAR regime cycles observed

### ⚙️ Technical Readiness

- [ ] Zero CRITICAL errors in last 5 trading days
- [ ] All Makefile commands execute cleanly
- [ ] Database backup procedure tested
- [ ] .env file secured (not in git)
- [ ] ANTHROPIC_API_KEY working and funded
- [ ] GROQ API key working (MiroFish simulation)
- [ ] ZEP_API_KEY working (agent memory)
- [ ] Telegram bot responds to all 10 commands

### 💰 Capital & Broker Setup

- [ ] Starting capital confirmed: NPR 500,000 minimum
- [ ] Broker account opened and funded
- [ ] DP (Demat) account active
- [ ] MEROSHARE account active
- [ ] Test order placed and cancelled successfully
- [ ] Commission schedule confirmed with broker

### 📋 Risk Management

- [ ] Daily loss limit alert set: -3%
- [ ] Weekly loss limit understood: -7%
- [ ] Monthly emergency stop set: -15%
- [ ] Stop-loss discipline tested (no exceptions)
- [ ] Position size limits understood

### 🔔 Monitoring & Alerts

- [ ] Morning health check (08:00 NST) tested
- [ ] Post-market summary (16:30 NST) tested
- [ ] Telegram manual order alerts tested
- [ ] Weekly review (Friday 18:00 NST) tested
- [ ] Price monitor stop-loss alerts tested

### 📝 Documentation

- [ ] This checklist completed
- [ ] Deployment readiness check run: `make deploy-readiness`
- [ ] Go-live check script run: `make go-live-check`
- [ ] Weekly review Week 1 through Week 4 saved
- [ ] Backtest report reviewed

---

## Go-Live Decision

- [ ] All blocking checks above passed
- [ ] `python -m deployment.readiness_check` exits with code 0
- [ ] `python -m deployment.go_live_check` confirms READY

**Date of go-live decision:** _______________

**Approved by:** _______________

**Starting capital (NPR):** _______________

**First live trade date:** _______________

---

## Post Go-Live (first 30 days)

- [ ] Week 1: Review performance vs paper trading expectations
- [ ] Week 2: Confirm stop-loss discipline maintained
- [ ] Week 3: Check signal accuracy vs paper trading baseline
- [ ] Week 4: First monthly review — consider capital scaling
