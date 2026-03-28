# Makefile — nepse-mirofish shortcuts
# Requires: make (Git Bash / WSL / chocolatey on Windows)

PYTHON := .venv/Scripts/python
PIP    := .venv/Scripts/pip
PYTEST := .venv/Scripts/pytest

.PHONY: install run test backfill db-reset db-migrate nrb seed sector health \
        mirofish-start mirofish-test agents simulate signal phase2 \
        strategy dashboard weekly-report phase3 \
        backtest backtest-full backtest-coverage backtest-seeds backtest-report \
        walk-forward regime-backtests stress-tests monte-carlo sensitivity \
        attribution optimise-agents deploy-check phase4 \
        paper-start paper-stop paper-status paper-run-now paper-fill-check \
        monitor-start bot-start dashboard-web weekly-review \
        deploy-readiness go-live-check phase5 paper-full

## Install all dependencies into the virtual environment
install:
	$(PIP) install -r requirements.txt

## Run the full daily pipeline immediately (skips scheduler wait)
run:
	$(PYTHON) scheduler/daily_pipeline.py --run-now

## Run pytest test suite
test:
	$(PYTEST) tests/ -v

## Run end-to-end health check (standalone, no pytest)
health:
	$(PYTHON) tests/test_pipeline_e2e.py

## Scrape NRB policy data now
nrb:
	$(PYTHON) scrapers/nrb_policy.py

## Build today's seed JSON
seed:
	$(PYTHON) pipeline/seed_builder.py

## Build sector summary from today's stocks data
sector:
	$(PYTHON) pipeline/sector_aggregator.py

## Run historical price backfill from 2022-01-01
backfill:
	$(PYTHON) -c "\
from db.loader import backfill_history; \
from scheduler.daily_pipeline import TOP_50_SYMBOLS; \
backfill_history(start_date='2022-01-01', symbols=TOP_50_SYMBOLS)"

## Create all tables directly (shortcut without Alembic)
db-reset:
	$(PYTHON) -c "\
from db.models import create_all_tables, get_engine, Base; \
engine = get_engine(); \
Base.metadata.drop_all(engine); \
print('All tables dropped.'); \
Base.metadata.create_all(engine); \
print('All tables recreated.')"

## Run Alembic migrations (requires PostgreSQL configured in .env)
db-migrate:
	.venv/Scripts/alembic upgrade head

## Generate a new Alembic migration (autogenerate from models)
db-revision:
	.venv/Scripts/alembic revision --autogenerate -m "$(MSG)"

## Start the scheduler (runs at 15:30 NST daily, Mon-Fri)
schedule:
	$(PYTHON) scheduler/daily_pipeline.py

## Verify environment setup
verify:
	$(PYTHON) verify_env.py

## ─── Phase 2: MiroFish Simulation ─────────────────────────────────────────

## Start MiroFish backend (localhost:5001) and frontend (localhost:3000)
mirofish-start:
	cd mirofish && npm run dev

## Test Claude LLM connection via MiroFish backend
mirofish-test:
	cd mirofish && ../.venv/Scripts/python test_llm_connection.py

## Show NEPSE agent config summary (1000 agents, 6 groups)
agents:
	$(PYTHON) pipeline/load_agents.py

## Run MiroFish simulation for today's seed
simulate:
	$(PYTHON) pipeline/run_simulation.py

## Extract signal from latest simulation transcript
signal:
	$(PYTHON) pipeline/signal_extractor.py --save

## Generate today's simulation question using Claude
question:
	$(PYTHON) pipeline/generate_simulation_question.py

## Initialise Zep agent memories (requires ZEP_API_KEY in .env)
init-memory:
	$(PYTHON) pipeline/init_agent_memory.py

## Run Phase 2 integration check (11 checks)
phase2:
	$(PYTHON) tests/test_phase2_integration.py

## ─── Phase 3: Strategy Layer ──────────────────────────────────────────────

## Run the full daily NEPSE strategy cycle (15 steps)
strategy:
	$(PYTHON) -m strategy.run_strategy

## Run strategy in dry-run mode (no paper trades placed)
strategy-dry:
	$(PYTHON) -m strategy.run_strategy --dry-run --skip-market-check

## Launch the rich terminal dashboard (live updating)
dashboard:
	$(PYTHON) -m strategy.dashboard

## Write HTML dashboard to data/dashboard.html (auto-refresh every 60s)
dashboard-html:
	$(PYTHON) -m strategy.dashboard --html --refresh 60

## Generate weekly Claude AI strategy report (uses ANTHROPIC_API_KEY)
weekly-report:
	$(PYTHON) -m strategy.report_generator

## Run Phase 3 integration tests
phase3:
	$(PYTEST) tests/test_phase3_integration.py -v

## Run ALL tests (Phase 1 + 2 + 3)
test-all:
	$(PYTEST) tests/ -v --tb=short

## Refresh event calendar cache
events:
	$(PYTHON) -c "from pipeline.event_calendar import save_event_cache; save_event_cache()"

## Create/migrate portfolio tables in DB
db-portfolio:
	$(PYTHON) -c "\
from db.models import get_engine, Base; \
from db.models import PortfolioPosition, TradeLog; \
engine = get_engine(); \
Base.metadata.create_all(engine, tables=[PortfolioPosition.__table__, TradeLog.__table__]); \
print('Portfolio tables created.')"

## ─── Phase 4: Backtesting & Validation ──────────────────────────────────────

## Check historical data coverage (2022-2024)
backtest-coverage:
	$(PYTHON) -m backtest.historical_seeds --check-coverage --start 2022-01-01 --end 2024-12-31

## Regenerate historical seeds (synthetic MiroFish signals for 2022-2024)
backtest-seeds:
	$(PYTHON) -m backtest.historical_seeds --regenerate --start 2022-01-01 --end 2024-12-31

## Run full backtest 2022-2024
backtest:
	$(PYTHON) -m backtest.engine --start 2022-01-01 --end 2024-12-31 --output results_full

## Run walk-forward validation (5 windows)
walk-forward:
	$(PYTHON) -m backtest.walk_forward --windows 5

## Run regime-specific backtests
regime-backtests:
	$(PYTHON) -m backtest.regime_backtests --all

## Run all 5 stress tests
stress-tests:
	$(PYTHON) -m backtest.stress_tests --all

## Run Monte Carlo simulation (5000 paths)
monte-carlo:
	$(PYTHON) -m backtest.monte_carlo --n 5000

## Run sensitivity analysis
sensitivity:
	$(PYTHON) -m backtest.sensitivity --all

## Run signal attribution analysis
attribution:
	$(PYTHON) -m backtest.signal_attribution

## Optimise MiroFish agent mix
optimise-agents:
	$(PYTHON) -m backtest.agent_optimiser --validate

## Generate comprehensive backtest report (md + 3 charts + json)
backtest-report:
	$(PYTHON) -m backtest.report

## Run deployment decision checker
deploy-check:
	$(PYTHON) -m backtest.deployment_decision

## Run Phase 4 integration tests
phase4:
	$(PYTEST) tests/test_phase4_complete.py tests/test_lookahead.py -v

## Run FULL Phase 4 pipeline (all steps in sequence)
backtest-full:
	$(MAKE) backtest-coverage
	$(MAKE) backtest-seeds
	$(MAKE) backtest
	$(MAKE) walk-forward
	$(MAKE) regime-backtests
	$(MAKE) stress-tests
	$(MAKE) monte-carlo
	$(MAKE) sensitivity
	$(MAKE) attribution
	$(MAKE) optimise-agents
	$(MAKE) backtest-report
	$(MAKE) deploy-check

## ─── Phase 5: Paper Trading & Live Deployment ────────────────────────────────

## Start paper trading scheduler (runs daily at 15:30 NST, Sun-Thu)
paper-start:
	$(PYTHON) -m paper_trading.engine

## Run paper trading cycle immediately (skip scheduler wait)
paper-run-now:
	$(PYTHON) -m paper_trading.engine --run-now

## Run morning fill check immediately
paper-fill-check:
	$(PYTHON) -m paper_trading.engine --fill-check

## Show current paper trading session status
paper-status:
	$(PYTHON) -m paper_trading.engine --status

## Stop paper trading (graceful shutdown)
paper-stop:
	@echo "Send SIGTERM to the paper trading process or press Ctrl+C"

## Launch Telegram bot
bot-start:
	$(PYTHON) -m paper_trading.telegram_bot

## Launch web dashboard at localhost:8080
dashboard-web:
	$(PYTHON) -m paper_trading.live_dashboard

## Run weekly review now (normally runs Friday 18:00 NST)
weekly-review:
	$(PYTHON) -m paper_trading.weekly_review --run-now

## Start production monitoring scheduler
monitor-start:
	$(PYTHON) -m deployment.monitor

## Run deployment readiness check
deploy-readiness:
	$(PYTHON) -m deployment.readiness_check

## Run go-live readiness check (comprehensive)
go-live-check:
	$(PYTHON) -m deployment.go_live_check

## Run Phase 5 integration tests
phase5:
	$(PYTEST) tests/test_full_system.py -v

## Run FULL Phase 5 setup sequence
paper-full:
	$(MAKE) db-portfolio
	$(MAKE) paper-run-now
	$(MAKE) bot-start &
	$(MAKE) dashboard-web &
	$(MAKE) monitor-start &
	@echo "Paper trading running. Check http://localhost:8080 and Telegram bot."

## ─────────────────────────────────────────────────────────────────────────────

## Show pipeline logs for today
logs:
	@$(PYTHON) -c "\
from pathlib import Path; from datetime import datetime; \
p = Path('logs') / f'pipeline_{datetime.now().strftime(\"%Y-%m-%d\")}.log'; \
print(p.read_text(encoding='utf-8') if p.exists() else 'No log for today.')"
