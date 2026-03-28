# NEPSE MiroFish — Manual Execution Guide

## Overview
This system operates in **MANUAL mode** for live trading on NEPSE.
The algorithm generates signals → you execute orders manually via your broker's platform.
The Telegram bot delivers exact order instructions to your phone.

## Trading Schedule (Nepal Standard Time)

| Time | Action |
|------|--------|
| 11:00 NST | NEPSE market opens |
| 11:15 NST | Morning fill check (system checks if limit orders should be placed) |
| 11:15–14:45 NST | Active trading window |
| 14:45 NST | Stop placing new orders (30 min before close) |
| 15:00 NST | NEPSE market closes |
| 15:30 NST | Daily cycle runs (signals generated for next day) |
| 16:30 NST | Post-market Telegram summary |
| 18:00 NST (Fri) | Weekly review published |

## Order Execution Workflow

### Step 1: Receive Order Alert
You will receive a Telegram message like:
```
🔔 ORDER ALERT — MANUAL EXECUTION REQUIRED
Action: BUY
Symbol: NABIL
Qty: 25 shares
Order Type: LIMIT
Limit Price: NPR 1,245.00
Estimated Value: NPR 31,125.00
Commission (0.425%): NPR 132.28
Net Cost: NPR 31,257.28
Rationale: Strong bull signal (score: 0.78)
⏰ Place before 14:45 NST today
Reply /confirm_BUY_NABIL_25 when executed
```

### Step 2: Log Into Your Broker Platform
- Recommended: TMS (Trading Management System) via your broker's website
- Or use your broker's mobile app

### Step 3: Place the Order
1. Navigate to Order Entry
2. Select symbol (e.g., NABIL)
3. Select order type: LIMIT
4. Enter quantity: 25
5. Enter limit price: 1,245.00
6. Review commission estimate
7. Submit order

### Step 4: Confirm Execution
Reply to the Telegram bot: `/confirm_BUY_NABIL_25`
The system logs the confirmation and updates the paper record.

### Step 5: Monitor During Trading Hours
Use `/positions` command to check your current positions.
The price monitor sends automatic alerts if a stop-loss is breached.

## Stop-Loss Execution

When you receive a stop-loss alert:
```
🚨 STOP-LOSS ALERT
Symbol: NICA
Current Price: NPR 852.00
Stop-Loss Level: NPR 860.00
Action: SELL IMMEDIATELY
Qty: 30 shares
```

**Execute immediately** — do not wait for the signal to "recover".
Stop-loss discipline is tracked and required for go-live.

## Settlement (T+3)

NEPSE uses T+3 settlement:
- Buy on Day 0 → Shares credited on Day 3
- Sell on Day 0 → Cash credited on Day 3
- You cannot sell shares before they are credited

## Capital Gains Tax

| Holding Period | CGT Rate |
|---------------|---------|
| < 365 days    | 7.5%    |
| ≥ 365 days    | 1.5%    |

CGT is deducted at source by your broker.
The system accounts for CGT in P&L calculations.

## Commission Structure

| Fee | Rate |
|-----|------|
| Broker commission | 0.40% |
| SEBON fee | 0.015% |
| DP charge | 0.010% |
| **Total** | **0.425%** |

Applied on both BUY and SELL transactions.

## Emergency Procedures

### If the system fails to generate a signal:
1. Check `make logs` for errors
2. Check Telegram for health alert
3. Run `make health` to diagnose
4. Do NOT trade until system is restored

### If NEPSE circuit breaker is triggered (±9.5%):
1. Do not place any new orders
2. Evaluate whether to exit existing positions
3. Wait for system's post-circuit analysis

### If you receive a /recalibrate command result:
Follow the recalibration instructions in the Telegram message.
Major recalibration = halt trading for 1 week.

## Makefile Quick Reference

```bash
make paper-start      # Start paper trading scheduler
make bot-start        # Start Telegram bot
make dashboard-web    # Open web dashboard at localhost:8080
make monitor-start    # Start production monitoring
make weekly-review    # Generate weekly review now
make paper-status     # Show current session status
make go-live-check    # Run go-live readiness check
```

## Contact & Support
This system was built for personal use on NEPSE.
All signals are AI-generated and should be used with your own judgment.
Never risk capital you cannot afford to lose.
