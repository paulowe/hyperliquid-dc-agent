import asyncio
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import httpx
import websockets
from hyperliquid.info import Info

# =========================
# Configuration
# =========================

load_dotenv()

WS_URL = os.getenv("HYPERLIQUID_PUBLIC_WS_URL")
BASE_URL = os.getenv("HYPERLIQUID_PUBLIC_BASE_URL")

LEADERS = [
    # 3rd by account value, 8th by PNL, 6th by volume
    "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00",
    # BobbyBigSize  6th by account value, 1st by PNL
    "0x7fdafde5cfb5465924316eced2d3715494c517d1",
    # x.com/silkbtc 3rd by PNL
    # 0x880ac484a1743862989a441d6d867238c7aa311c
]

POLL_SECONDS = 60
HEARTBEAT_SECONDS = 25  # keep < 60s idle timeout (docs)
SWITCH_COOLDOWN_SECONDS = 6 * 60 * 60  # 6 hours
HYSTERESIS_RETURN = 0.01  # 1% improvement required to switch
MAX_DD_PENALTY = 0.02  # prefer lower drawdown by 2%+

# =========================
# Data Model
# =========================

@dataclass
class LeaderState:
    addr: str
    fills: List[dict] = field(default_factory=list)
    equity: List[Tuple[float, float]] = field(default_factory=list)  # (ts, equity_usdc)
    last_eq_ts: float = 0.0  # last appended timestamp (seconds)

    def append_equity(self, ts: float, eq: float) -> None:
        # protect against out-of-order or duplicate timestamps
        if ts <= self.last_eq_ts:
            return
        self.equity.append((ts, eq))
        self.last_eq_ts = ts

        # keep last 48h
        cutoff = ts - 48 * 3600
        while self.equity and self.equity[0][0] < cutoff:
            self.equity.pop(0)


# =========================
# Portfolio Parsing (SDK)
# =========================

def portfolio_pairs_to_dict(portfolio_resp) -> dict:
    """
    Convert:
      [['day', {...}], ['week', {...}], ...]
    into:
      {'day': {...}, 'week': {...}, ...}
    """
    if not isinstance(portfolio_resp, list):
        return {}
    out = {}
    for item in portfolio_resp:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
            out[item[0]] = item[1]
    return out

def extract_account_value_history(portfolio_resp, bucket: str = "day") -> List[Tuple[float, float]]:
    """
    Returns a list of (ts_seconds, equity_float) from accountValueHistory.
    Handles strings and ms timestamps.
    """
    d = portfolio_pairs_to_dict(portfolio_resp)
    b = d.get(bucket) or {}
    hist = b.get("accountValueHistory") or []

    pts: List[Tuple[float, float]] = []
    for row in hist:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        ts_raw, v_raw = row[0], row[1]
        try:
            ts = float(ts_raw)
            v = float(v_raw)
        except Exception:
            continue

        # Hyperliquid often uses ms timestamps in some endpoints; normalize if needed
        if ts > 1e12:  # looks like ms
            ts = ts / 1000.0

        pts.append((ts, v))

    pts.sort(key=lambda x: x[0])
    return pts

def extract_latest_equity_usdc(portfolio_resp, bucket: str = "day") -> Optional[float]:
    pts = extract_account_value_history(portfolio_resp, bucket=bucket)
    if not pts:
        return None
    return pts[-1][1]


# =========================
# Metrics
# =========================

def metrics_from_equity(equity: List[Tuple[float, float]]) -> dict:
    if len(equity) < 3:
        return {"ret_1h": 0.0, "ret_24h": 0.0, "max_dd": 0.0, "vol": 0.0, "pos_pct": 0.0}

    # returns per interval
    vals = [v for _, v in equity]
    rets = []
    pos = 0
    for i in range(1, len(vals)):
        if vals[i-1] <= 0:
            continue
        r = (vals[i] / vals[i-1]) - 1.0
        rets.append(r)
        if r > 0:
            pos += 1

    pos_pct = pos / max(1, len(rets))

    # max drawdown
    peak = -1e18
    max_dd = 0.0
    for v in vals:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

    # volatility of interval returns
    if len(rets) >= 2:
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
        vol = math.sqrt(var)
    else:
        vol = 0.0

    # helper to compute return over window
    def window_return(window_s: int) -> float:
        now_ts = equity[-1][0]
        now_v = equity[-1][1]
        target_ts = now_ts - window_s
        # find oldest point >= target_ts
        idx = 0
        while idx < len(equity) and equity[idx][0] < target_ts:
            idx += 1
        if idx == len(equity):
            base_v = equity[0][1]
        else:
            base_v = equity[idx][1]
        if base_v <= 0:
            return 0.0
        return (now_v / base_v) - 1.0

    return {
        "ret_1h": window_return(3600),
        "ret_24h": window_return(24 * 3600),
        "max_dd": max_dd,
        "vol": vol,
        "pos_pct": pos_pct,
    }

def score(m: dict) -> float:
    # simple “quality” score: reward return + consistency, penalize drawdown + vol
    return (m["ret_24h"] * 1.0) + (m["pos_pct"] * 0.1) - (m["max_dd"] * 0.8) - (m["vol"] * 0.2)

# =========================
# Portfolio Polling (SDK)
# =========================

async def poll_portfolio(info: Info, leader: LeaderState, stop: asyncio.Event, bucket: str = "day"):
    """
    Pull the whole history series for the bucket, then append only new timestamps
    "day" is usually most responsive for “recent performance”.
    "perpDay" is perps-only (if you want to ignore spot).
    Your scoring uses ret_24h, so "day" is the natural match.
    """

    while not stop.is_set():
        try:
            data = await asyncio.to_thread(info.portfolio, leader.addr)

            # Pull the whole history series for the bucket, then append only new timestamps
            pts = extract_account_value_history(data, bucket=bucket)

            if pts:
                # Append only points newer than the last seen timestamp
                for ts, eq in pts:
                    leader.append_equity(ts, eq)

        except Exception as e:
            print(f"[{leader.addr}] portfolio error: {e}")

        await asyncio.sleep(POLL_SECONDS)


# =========================
# WebSocket Listener (Reconnect + Heartbeat)
# =========================

async def ws_fills_listener(leaders: Dict[str, LeaderState], stop: asyncio.Event):
    while not stop.is_set():
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,
                close_timeout=5,
            ) as ws:
                # subscribe to each leader's userEvents
                for addr in leaders:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "userEvents", "user": addr},
                    }))
                
                async def heartbeat():
                    while not stop.is_set():
                        try:
                            await ws.send(json.dumps({"method": "ping"}))
                        except Exception:
                            break
                        await asyncio.sleep(HEARTBEAT_SECONDS)

                hb_task = asyncio.create_task(heartbeat())

                async for msg in ws:
                    if stop.is_set():
                        break
                    try:
                        d = json.loads(msg)
                        if d.get("channel") == "pong":
                            continue
                        if d.get("channel") not in ("user", "userEvents"):
                            continue
                        payload = d.get("data", {})
                        fills = payload.get("fills", [])
                        # some WS payloads include user address; if not, you may need
                        # separate connections per leader. We'll attempt best-effort:
                        user = payload.get("user") or payload.get("address")  # may be None
                        if user in leaders:
                            leaders[user].fills.extend(fills)
                    except Exception:
                        continue

                hb_task.cancel()
        except Exception as e:
            print(f"[ws] disconnected: {e} (reconnecting...)")
            await asyncio.sleep(1.5)

# =========================
# Leader Chooser
# =========================

async def chooser(leaders: Dict[str, LeaderState], stop: asyncio.Event):
    current = None
    last_switch = 0.0

    while not stop.is_set():
        scored = []
        for addr, st in leaders.items():
            m = metrics_from_equity(st.equity)
            s = score(m)
            scored.append((s, addr, m))

        scored.sort(reverse=True, key=lambda x: x[0])
        if not scored:
            await asyncio.sleep(5)
            continue

        best_s, best_addr, best_m = scored[0]
        if current is None:
            current = best_addr
            last_switch = time.time()
            print(f"✅ initial leader = {current[:8]}  metrics={best_m}")
        else:
            cur_state = leaders[current]
            cur_m = metrics_from_equity(cur_state.equity)
            cur_s = score(cur_m)

            now = time.time()
            cooldown_ok = (now - last_switch) >= SWITCH_COOLDOWN_SECONDS

            # switch only if materially better AND drawdown advantage
            if cooldown_ok:
                if (best_s > cur_s + HYSTERESIS_RETURN) and (best_m["max_dd"] + MAX_DD_PENALTY < cur_m["max_dd"]):
                    print(f"🔁 switching leader {current[:8]} -> {best_addr[:8]}")
                    print(f"   current: {cur_m}")
                    print(f"   best:    {best_m}")
                    current = best_addr
                    last_switch = now

        # periodic print
        if scored:
            top3 = ", ".join([f"{a[:8]} s={s:.4f} dd={m['max_dd']:.2%} r24={m['ret_24h']:.2%}" for s, a, m in scored[:3]])
            print(f"🏁 top: {top3}")

        await asyncio.sleep(15)

# =========================
# Main
# =========================

async def main():
    stop = asyncio.Event()

    def on_sigint(signum, frame):
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    leaders = {addr: LeaderState(addr=addr) for addr in LEADERS}
    info = Info(BASE_URL, skip_ws=True)

    tasks = [
        asyncio.create_task(ws_fills_listener(leaders, stop)),
        asyncio.create_task(chooser(leaders, stop)),
    ]
    
    for st in leaders.values():
        tasks.append(asyncio.create_task(poll_portfolio(info, st, stop, bucket="day")))


    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
