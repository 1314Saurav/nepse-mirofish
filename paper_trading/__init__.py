"""
paper_trading/ — NEPSE MiroFish Phase 5: Paper Trading & Live Deployment.
"""
# Auto-load .env for every paper_trading module
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
