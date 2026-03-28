"""
paper_trading/signal_tracker.py
Real-time signal accuracy tracking for paper trading.

For every signal generated during paper trading, tracks whether the
predicted market direction actually materialised.
Evaluation windows: 1, 3, 5, 10 trading days after signal.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SignalAccuracyTracker:
    """
    Records and evaluates signal accuracy in real time.
    Persists to disk so history survives restarts.
    """

    EVAL_WINDOWS = [1, 3, 5, 10]  # trading days

    def __init__(self, session_id: str, data_dir: str = "data/paper_trading"):
        self.session_id = session_id
        self._dir = Path(data_dir) / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._signals_path = self._dir / "signal_history.json"
        self._accuracy_path = self._dir / "accuracy_report.json"
        self._signals: list[dict] = self._load_signals()

    # ── Recording ───────────────────────────────────────────────────────────

    def record_signal(self, date_str: str, signal: dict) -> None:
        """Record today's composite signal for future evaluation."""
        # Avoid duplicates
        existing_dates = {s["date"] for s in self._signals}
        if date_str in existing_dates:
            logger.debug("Signal for %s already recorded", date_str)
            return

        mf_score = signal.get("mirofish_score", 0.0)
        if isinstance(mf_score, dict):
            mf_score = mf_score.get("bull_bear_score", 0.0)

        record = {
            "date": date_str,
            "mirofish_score": float(mf_score),
            "regime": signal.get("regime", "UNKNOWN"),
            "action": signal.get("action", "HOLD"),
            "outcomes": {},      # filled in by evaluate_past_signals
            "evaluated": False,
        }

        # Extract per-symbol composite scores if available
        composite = signal.get("composite", {})
        if isinstance(composite, dict):
            record["top_buys"] = [
                sym for sym, s in composite.items()
                if isinstance(s, dict) and s.get("action") == "BUY"
            ][:5]

        self._signals.append(record)
        self._save_signals()
        logger.info("Signal recorded for %s: score=%.3f action=%s",
                    date_str, float(mf_score), record["action"])

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate_past_signals(self, market_data: dict) -> list[dict]:
        """
        For signals generated 1, 3, 5, 10 trading days ago:
        - Was BULLISH signal followed by NEPSE gain?
        - Was BEARISH signal followed by NEPSE decline?
        Returns list of newly evaluated signals.
        """
        today = date.today()
        newly_evaluated = []

        try:
            from backtest.calendar import get_trading_days
        except ImportError:
            get_trading_days = None

        nepse_today = market_data.get("nepse_index", 0.0) if isinstance(market_data, dict) else 0.0

        for record in self._signals:
            if record.get("evaluated"):
                continue

            signal_date_str = record["date"]
            try:
                signal_date = date.fromisoformat(signal_date_str)
            except ValueError:
                continue

            # Compute trading days since this signal
            if get_trading_days:
                try:
                    days_since = len(get_trading_days(signal_date_str, str(today))) - 1
                except Exception:
                    days_since = max(0, (today - signal_date).days)
            else:
                days_since = max(0, (today - signal_date).days)

            if days_since < 1:
                continue

            # Try to get market returns for each evaluation window
            outcomes = {}
            for window in self.EVAL_WINDOWS:
                if days_since < window:
                    continue
                # Simplistic evaluation using today's data if window days have passed
                try:
                    nepse_at_signal = self._get_nepse_on_date(signal_date_str)
                    nepse_at_target = self._get_nepse_n_days_later(signal_date_str, window)
                    if nepse_at_signal and nepse_at_target:
                        nepse_ret = (nepse_at_target / nepse_at_signal - 1) * 100
                        mf = record.get("mirofish_score", 0)
                        predicted_bull = mf > 0.1
                        predicted_bear = mf < -0.1
                        actual_bull = nepse_ret > 0
                        correct = (predicted_bull and actual_bull) or (predicted_bear and not actual_bull)
                        outcomes[f"d{window}"] = {
                            "nepse_return_pct": round(nepse_ret, 3),
                            "predicted": "BULL" if predicted_bull else ("BEAR" if predicted_bear else "NEUTRAL"),
                            "actual": "BULL" if actual_bull else "BEAR",
                            "correct": correct,
                        }
                except Exception as exc:
                    logger.debug("Outcome eval failed for %s d%d: %s", signal_date_str, window, exc)

            if outcomes:
                record["outcomes"] = outcomes
                max_eval = max(int(k[1:]) for k in outcomes.keys())
                if max_eval >= max(self.EVAL_WINDOWS) or days_since >= max(self.EVAL_WINDOWS):
                    record["evaluated"] = True
                newly_evaluated.append(record)

        if newly_evaluated:
            self._save_signals()
            self._update_accuracy_report()

        return newly_evaluated

    def _get_nepse_on_date(self, date_str: str) -> Optional[float]:
        """Fetch NEPSE index for a specific date from DB."""
        try:
            from db.loader import get_market_snapshot
            snap = get_market_snapshot(date_str)
            if snap:
                return snap.get("nepse_index") or snap.get("nepse_close")
        except Exception:
            pass
        return None

    def _get_nepse_n_days_later(self, date_str: str, n: int) -> Optional[float]:
        """Fetch NEPSE index n trading days after date_str."""
        try:
            from backtest.calendar import get_trading_days
            all_days = get_trading_days(date_str, str(date.today()))
            if len(all_days) > n:
                target_date = all_days[n]
                return self._get_nepse_on_date(str(target_date))
        except Exception:
            pass
        return None

    # ── Accuracy report ─────────────────────────────────────────────────────

    def get_accuracy_report(self) -> dict:
        """
        Compute running accuracy statistics from all evaluated signals.
        """
        evaluated = [s for s in self._signals if s.get("outcomes")]
        if not evaluated:
            return {
                "total_signals_evaluated": 0,
                "message": "No signals evaluated yet",
                "bull_accuracy_1d": None,
                "bull_accuracy_3d": None,
                "bull_accuracy_5d": None,
            }

        def _accuracy(window_key: str, direction: str) -> Optional[float]:
            """% of signals where direction prediction was correct."""
            relevant = [
                s for s in evaluated
                if window_key in s.get("outcomes", {})
                and s["outcomes"][window_key].get("predicted") == direction
            ]
            if not relevant:
                return None
            correct = sum(1 for s in relevant if s["outcomes"][window_key].get("correct"))
            return round(correct / len(relevant) * 100, 1)

        def _accuracy_any(window_key: str) -> Optional[float]:
            """Overall directional accuracy for a window (bull + bear combined)."""
            relevant = [
                s for s in evaluated
                if window_key in s.get("outcomes", {})
                and s["outcomes"][window_key].get("predicted") != "NEUTRAL"
            ]
            if not relevant:
                return None
            correct = sum(1 for s in relevant if s["outcomes"][window_key].get("correct"))
            return round(correct / len(relevant) * 100, 1)

        # Watchlist accuracy: did BUY-rated stocks rise in 5d?
        watchlist_hits = 0
        watchlist_total = 0
        for s in evaluated:
            if "d5" in s.get("outcomes", {}) and s.get("top_buys"):
                outcome = s["outcomes"]["d5"]
                watchlist_total += 1
                if outcome.get("actual") == "BULL":
                    watchlist_hits += 1
        watchlist_acc = round(watchlist_hits / watchlist_total * 100, 1) if watchlist_total > 0 else None

        report = {
            "total_signals_evaluated": len(evaluated),
            "total_signals_recorded": len(self._signals),
            "bull_accuracy_1d": _accuracy("d1", "BULL"),
            "bull_accuracy_3d": _accuracy("d3", "BULL"),
            "bull_accuracy_5d": _accuracy("d5", "BULL"),
            "bear_accuracy_1d": _accuracy("d1", "BEAR"),
            "bear_accuracy_3d": _accuracy("d3", "BEAR"),
            "bear_accuracy_5d": _accuracy("d5", "BEAR"),
            "overall_accuracy_3d": _accuracy_any("d3"),
            "overall_accuracy_5d": _accuracy_any("d5"),
            "watchlist_accuracy_5d": watchlist_acc,
            "last_10_signals_accuracy": self._last_n_accuracy(10),
            "accuracy_trend": self._accuracy_trend(),
        }

        # Regime breakdown
        regime_acc: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for s in evaluated:
            regime = s.get("regime", "UNKNOWN")
            if "d3" in s.get("outcomes", {}):
                oc = s["outcomes"]["d3"]
                if oc.get("predicted") != "NEUTRAL":
                    regime_acc[regime]["total"] += 1
                    if oc.get("correct"):
                        regime_acc[regime]["correct"] += 1
        report["regime_accuracy_3d"] = {
            r: round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else None
            for r, v in regime_acc.items()
        }

        return report

    def _last_n_accuracy(self, n: int) -> Optional[float]:
        """Accuracy of last n evaluated signals at 3d window."""
        recent = [s for s in self._signals if "d3" in s.get("outcomes", {})][-n:]
        if not recent:
            return None
        correct = sum(1 for s in recent
                      if s["outcomes"]["d3"].get("predicted") != "NEUTRAL"
                      and s["outcomes"]["d3"].get("correct"))
        directional = sum(1 for s in recent
                          if s["outcomes"]["d3"].get("predicted") != "NEUTRAL")
        return round(correct / directional * 100, 1) if directional > 0 else None

    def _accuracy_trend(self) -> str:
        """Is accuracy improving, stable, or declining?"""
        recent_5 = self._last_n_accuracy(5)
        recent_15 = self._last_n_accuracy(15)
        if recent_5 is None or recent_15 is None:
            return "insufficient_data"
        diff = (recent_5 or 0) - (recent_15 or 0)
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        return "stable"

    def generate_accuracy_alert(self) -> str:
        """
        Format a weekly accuracy Telegram message.
        Flags if accuracy drops below 50% over last 10 signals.
        """
        report = self.get_accuracy_report()
        last10 = report.get("last_10_signals_accuracy")
        trend = report.get("accuracy_trend", "unknown")
        total = report.get("total_signals_evaluated", 0)

        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "❓")

        lines = [
            "📊 *MiroFish Signal Accuracy Report*",
            "",
            f"Evaluated: {total} signals",
            f"Trend: {trend_emoji} {trend.upper()}",
            "",
            "*3-Day Accuracy:*",
        ]

        bull3 = report.get("bull_accuracy_3d")
        bear3 = report.get("bear_accuracy_3d")
        overall3 = report.get("overall_accuracy_3d")

        if bull3 is not None:
            lines.append(f"  Bullish: `{bull3:.0f}%`")
        if bear3 is not None:
            lines.append(f"  Bearish: `{bear3:.0f}%`")
        if overall3 is not None:
            lines.append(f"  Overall: `{overall3:.0f}%`")

        if last10 is not None:
            flag = " ⚠️ BELOW 50% — CHECK REGIME SHIFT" if last10 < 50 else ""
            lines.append(f"\nLast 10 signals: `{last10:.0f}%`{flag}")

        watch5 = report.get("watchlist_accuracy_5d")
        if watch5 is not None:
            lines.append(f"Watchlist (5d): `{watch5:.0f}%`")

        lines.append(f"\n_Generated {date.today()}_")
        return "\n".join(lines)

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load_signals(self) -> list[dict]:
        if self._signals_path.exists():
            try:
                with open(self._signals_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return []

    def _save_signals(self) -> None:
        with open(self._signals_path, "w", encoding="utf-8") as fh:
            json.dump(self._signals, fh, indent=2, default=str)

    def _update_accuracy_report(self) -> None:
        """Recompute and save accuracy report to disk."""
        report = self.get_accuracy_report()
        with open(self._accuracy_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Accuracy report updated: overall_3d=%s%%",
                    report.get("overall_accuracy_3d"))
