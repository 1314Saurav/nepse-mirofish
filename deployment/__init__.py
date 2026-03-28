"""
deployment/ — NEPSE MiroFish Phase 5: Live Deployment Layer

Modules:
  readiness_check   – go/no-go criteria before deploying real capital
  risk_management   – live trading risk rules and position sizing
  broker_prep       – broker integration layer (PAPER → MANUAL → API)
  monitor           – production health monitoring
  recalibration     – when and how to recalibrate the strategy
  go_live_check     – auto-fill the GO_LIVE_CHECKLIST
"""
