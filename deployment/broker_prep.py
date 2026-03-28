"""
deployment/broker_prep.py
Broker integration layer for NEPSE live trading.

Three execution modes:
  PAPER  — simulated execution (default during paper trading)
  MANUAL — Telegram alerts with exact order instructions for human execution
  API    — direct broker API (future, not yet implemented for NEPSE)

NEPSE does not have a retail algorithmic trading API.
Manual execution mode is the primary live deployment method.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nepal Standard Time offset (UTC+5:45)
# ---------------------------------------------------------------------------
NST_OFFSET = timedelta(hours=5, minutes=45)


def _nst_now() -> datetime:
    """Return current time as NST (UTC+5:45)."""
    nst = timezone(NST_OFFSET)
    return datetime.now(tz=nst)


# ---------------------------------------------------------------------------
# NEPSE broker information constants
# ---------------------------------------------------------------------------

NEPSE_BROKER_INFO: dict = {
    "settlement": "T+3",
    "commission": {
        "broker_pct": 0.40,
        "sebon_pct": 0.015,
        "dp_pct": 0.010,
        "total_approx_pct": 0.425,
        "description": "0.4% broker + 0.015% SEBON + 0.01% DP = ~0.425% total",
    },
    "capital_gains_tax": {
        "short_term_pct": 7.5,
        "long_term_pct": 1.5,
        "long_term_threshold_days": 365,
        "description": "7.5% on profits if held ≤ 365 days; 1.5% if held > 365 days",
    },
    "market_hours": {
        "open": "11:00 NST",
        "close": "15:00 NST",
        "timezone": "Asia/Kathmandu (UTC+5:45)",
        "recommended_order_cutoff": "14:45 NST",
    },
    "order_types": ["LIMIT", "MARKET (AT_MARKET)"],
    "trading_days": "Sunday–Thursday",
    "notes": [
        "NEPSE has no retail algorithmic trading API as of 2025.",
        "All live orders must be placed via broker trading portal.",
        "MEROSHARE is used for DP (Demat) account management.",
    ],
}

# ---------------------------------------------------------------------------
# Execution modes registry
# ---------------------------------------------------------------------------

EXECUTION_MODES: dict = {
    "PAPER": {
        "description": "Simulated execution — no real capital at risk.",
        "use_case": "Paper trading period (minimum 20 trading days before live capital).",
        "order_routing": "paper_trading.engine.PaperTradingEngine.simulate_order()",
        "telegram_alert": False,
        "real_money": False,
    },
    "MANUAL": {
        "description": "Telegram alerts with exact order instructions for human execution.",
        "use_case": "Primary live deployment method for NEPSE (no retail API available).",
        "order_routing": "Human via broker trading portal after receiving Telegram alert.",
        "telegram_alert": True,
        "real_money": True,
    },
    "API": {
        "description": "Direct broker API integration (not yet available for NEPSE retail).",
        "use_case": "Future use — reserved for when NEPSE opens algorithmic trading access.",
        "order_routing": "NotImplemented — raises NotImplementedError at runtime.",
        "telegram_alert": False,
        "real_money": True,
    },
}


# ---------------------------------------------------------------------------
# BrokerIntegrationLayer
# ---------------------------------------------------------------------------

class BrokerIntegrationLayer:
    """
    Abstraction layer for order routing across three execution modes.

    In PAPER mode orders are forwarded to the paper trading engine.
    In MANUAL mode a Telegram alert is sent with full execution instructions.
    In API mode a NotImplementedError is raised (NEPSE API not yet available).

    Parameters
    ----------
    mode : str
        One of "PAPER", "MANUAL", or "API". Defaults to "PAPER".
    telegram_chat_id : str | None
        Telegram chat ID override. Falls back to TELEGRAM_CHAT_ID env var.
    """

    def __init__(
        self,
        mode: str = "PAPER",
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        if mode not in EXECUTION_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {list(EXECUTION_MODES)}"
            )
        self.mode = mode
        self.telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._orders_placed: int = 0
        self._orders_filled: int = 0
        self._orders_missed: int = 0
        self._order_log: list[dict] = []
        logger.info("BrokerIntegrationLayer initialised in %s mode.", mode)

    # ── Public API ──────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_type: str = "LIMIT",
        limit_price: Optional[float] = None,
        rationale: str = "",
    ) -> dict:
        """
        Route an order according to the active execution mode.

        Parameters
        ----------
        symbol : str
            NEPSE ticker (e.g. "NABIL", "EBL").
        action : str
            "BUY" or "SELL".
        qty : int
            Number of shares.
        order_type : str
            "LIMIT" or "MARKET".
        limit_price : float | None
            Required for LIMIT orders.
        rationale : str
            Human-readable reason for the order (included in alerts).

        Returns
        -------
        dict
            Order result with status, mode, and routing details.
        """
        logger.info(
            "place_order: mode=%s  %s %d %s @ %s",
            self.mode, action, qty, symbol, order_type,
        )

        result: dict = {
            "mode": self.mode,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "rationale": rationale,
            "timestamp": _nst_now().isoformat(),
        }

        if self.mode == "PAPER":
            result.update(self._route_paper(symbol, action, qty, order_type, limit_price, rationale))

        elif self.mode == "MANUAL":
            alert_text = self._send_manual_execution_alert(
                symbol=symbol,
                action=action,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price,
                rationale=rationale,
            )
            result["status"] = "ALERT_SENT"
            result["alert_preview"] = alert_text[:200] + "..." if len(alert_text) > 200 else alert_text

        elif self.mode == "API":
            raise NotImplementedError("NEPSE API not yet available for retail")

        self._orders_placed += 1
        self._order_log.append(result)
        return result

    def _route_paper(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_type: str,
        limit_price: Optional[float],
        rationale: str,
    ) -> dict:
        """Forward to the paper trading engine's simulate_order."""
        try:
            from paper_trading.engine import PaperTradingEngine
            # Load the most-recent live engine state if one exists on disk.
            engine = PaperTradingEngine()
            order_result = engine.simulate_order(
                symbol=symbol,
                action=action,
                qty=qty,
                price_type=order_type,
                limit_price=limit_price,
                signal_context={"rationale": rationale},
            )
            return {"status": "PAPER_PENDING", "paper_order": order_result}
        except Exception as exc:
            logger.warning("Paper routing failed: %s", exc)
            return {"status": "PAPER_ERROR", "error": str(exc)}

    # ── Manual execution alert ──────────────────────────────────────────────

    def _send_manual_execution_alert(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_type: str,
        limit_price: Optional[float],
        rationale: str,
    ) -> str:
        """
        Format and send a Telegram execution alert.

        Returns the formatted message string (for logging / testing).
        """
        price = limit_price or 0.0
        estimated_value = price * qty
        commission_detail = self.calculate_commission(estimated_value)
        commission_total = commission_detail["total"]
        net_cost = commission_detail["net_cost"] if action == "BUY" else estimated_value - commission_total

        cutoff_time = "14:45 NST"
        confirm_cmd = f"/confirm_{action}_{symbol}_{qty}"

        lines = [
            "🔔 ORDER ALERT — MANUAL EXECUTION REQUIRED",
            "══════════════════════════════",
            f"Action: {action}",
            f"Symbol: {symbol}",
            f"Qty: {qty} shares",
            f"Order Type: {order_type}",
            f"Limit Price: NPR {price:,.2f}" if limit_price else "Limit Price: AT MARKET",
            f"Estimated Value: NPR {estimated_value:,.2f}",
            f"Commission (0.425%): NPR {commission_total:,.2f}",
            f"Net Cost: NPR {net_cost:,.2f}",
            "──────────────────────────────",
            f"Rationale: {rationale}",
            "──────────────────────────────",
            f"⏰ Place before {cutoff_time} today",
            f"Reply {confirm_cmd} when executed",
        ]
        message = "\n".join(lines)

        self._telegram_send(message)
        logger.info("Manual execution alert sent for %s %s %d.", action, symbol, qty)
        return message

    # ── Commission & tax helpers ────────────────────────────────────────────

    def calculate_commission(self, value_npr: float) -> dict:
        """
        Break down NEPSE transaction costs on a given trade value.

        Parameters
        ----------
        value_npr : float
            Gross trade value in NPR.

        Returns
        -------
        dict
            Keys: broker, sebon, dp, total, net_cost.
        """
        broker = value_npr * (NEPSE_BROKER_INFO["commission"]["broker_pct"] / 100)
        sebon = value_npr * (NEPSE_BROKER_INFO["commission"]["sebon_pct"] / 100)
        dp = value_npr * (NEPSE_BROKER_INFO["commission"]["dp_pct"] / 100)
        total = broker + sebon + dp
        net_cost = value_npr + total
        return {
            "broker": round(broker, 2),
            "sebon": round(sebon, 2),
            "dp": round(dp, 2),
            "total": round(total, 2),
            "net_cost": round(net_cost, 2),
        }

    def calculate_cgt(
        self,
        buy_price: float,
        sell_price: float,
        qty: int,
        hold_days: int,
    ) -> dict:
        """
        Compute Nepal capital gains tax on a completed trade.

        Parameters
        ----------
        buy_price : float
            Average purchase price per share (NPR).
        sell_price : float
            Sale price per share (NPR).
        qty : int
            Number of shares sold.
        hold_days : int
            Number of calendar days the position was held.

        Returns
        -------
        dict
            Keys: cgt_rate, cgt_amount, net_proceeds.
        """
        cgt_info = NEPSE_BROKER_INFO["capital_gains_tax"]
        gross_proceeds = sell_price * qty
        cost_basis = buy_price * qty
        gross_profit = gross_proceeds - cost_basis

        if gross_profit <= 0:
            cgt_rate = 0.0
            cgt_amount = 0.0
        elif hold_days > cgt_info["long_term_threshold_days"]:
            cgt_rate = cgt_info["long_term_pct"] / 100
            cgt_amount = gross_profit * cgt_rate
        else:
            cgt_rate = cgt_info["short_term_pct"] / 100
            cgt_amount = gross_profit * cgt_rate

        sell_commission = self.calculate_commission(gross_proceeds)["total"]
        net_proceeds = gross_proceeds - sell_commission - cgt_amount

        return {
            "cgt_rate": cgt_rate,
            "cgt_rate_pct": cgt_rate * 100,
            "gross_profit": round(gross_profit, 2),
            "cgt_amount": round(cgt_amount, 2),
            "sell_commission": round(sell_commission, 2),
            "net_proceeds": round(net_proceeds, 2),
            "hold_days": hold_days,
            "long_term": hold_days > cgt_info["long_term_threshold_days"],
        }

    # ── Execution summary ───────────────────────────────────────────────────

    def get_execution_summary(self) -> dict:
        """
        Return statistics on orders placed via this layer.

        Returns
        -------
        dict
            Keys: mode, orders_placed, orders_filled, orders_missed,
            fill_rate_pct, order_log_tail.
        """
        fill_rate = (
            self._orders_filled / self._orders_placed * 100
            if self._orders_placed > 0 else 0.0
        )
        return {
            "mode": self.mode,
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_missed": self._orders_missed,
            "fill_rate_pct": round(fill_rate, 1),
            "order_log_tail": self._order_log[-10:],
        }

    def mark_order_filled(self, symbol: str) -> None:
        """Record that a manually confirmed order was filled."""
        self._orders_filled += 1
        logger.info("Order marked filled: %s", symbol)

    def mark_order_missed(self, symbol: str) -> None:
        """Record that an order was not executed."""
        self._orders_missed += 1
        logger.info("Order marked missed: %s", symbol)

    # ── Internal Telegram sender ────────────────────────────────────────────

    def _telegram_send(self, message: str) -> bool:
        """Send a plain-text message via Telegram Bot API."""
        import urllib.request
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = self.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot_token or not chat_id or "your_" in bot_token:
            logger.debug("Telegram not configured — alert logged only.")
            return False
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message[:4096],
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NEPSE MiroFish broker integration layer")
    parser.add_argument("--mode", choices=["PAPER", "MANUAL", "API"], default="PAPER",
                        help="Execution mode (default: PAPER)")
    parser.add_argument("--demo-order", action="store_true",
                        help="Send a demo MANUAL order alert and exit")
    parser.add_argument("--commission", type=float, metavar="VALUE_NPR",
                        help="Calculate commission breakdown for a trade value")
    parser.add_argument("--cgt", nargs=4, metavar=("BUY", "SELL", "QTY", "DAYS"),
                        help="Calculate CGT: --cgt <buy_price> <sell_price> <qty> <hold_days>")
    args = parser.parse_args()

    layer = BrokerIntegrationLayer(mode=args.mode)

    if args.demo_order:
        result = layer.place_order(
            symbol="NABIL",
            action="BUY",
            qty=25,
            order_type="LIMIT",
            limit_price=1245.00,
            rationale="Strong bull signal (score: 0.78), Banking sector momentum",
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.commission is not None:
        comm = layer.calculate_commission(args.commission)
        print(json.dumps(comm, indent=2))

    elif args.cgt:
        buy_p, sell_p, qty, days = args.cgt
        cgt = layer.calculate_cgt(float(buy_p), float(sell_p), int(qty), int(days))
        print(json.dumps(cgt, indent=2))

    else:
        print("NEPSE Broker Integration Layer")
        print(f"  Active mode : {layer.mode}")
        print(f"  Settlement  : {NEPSE_BROKER_INFO['settlement']}")
        print(f"  Commission  : ~{NEPSE_BROKER_INFO['commission']['total_approx_pct']}%")
        print(f"  CGT         : {NEPSE_BROKER_INFO['capital_gains_tax']['short_term_pct']}% "
              f"(short) / {NEPSE_BROKER_INFO['capital_gains_tax']['long_term_pct']}% (long)")
        print(f"  Market hours: {NEPSE_BROKER_INFO['market_hours']['open']} – "
              f"{NEPSE_BROKER_INFO['market_hours']['close']}")
        print(f"  Trading days: {NEPSE_BROKER_INFO['trading_days']}")
        print("\nUse --demo-order to send a sample manual alert.")


if __name__ == "__main__":
    main()
