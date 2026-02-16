"""
Simple raw message printer for ALL WebSocket messages.
Shows unprocessed JSON messages from the API including positions, fills, and orders.
"""

import asyncio
import json
import os
import signal
from dotenv import load_dotenv
import websockets

load_dotenv()

# WS_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_WS_URL")
WS_URL = os.getenv("HYPERLIQUID_PUBLIC_WS_URL")
# LEADER_ADDRESS = os.getenv("TESTNET_WALLET_ADDRESS")
LEADER_ADDRESS = '0xd47587702a91731dc1089b5db0932cf820151a91'
PING_EVERY = 25  # must be < 60s; 

async def hl_heartbeat(ws: websockets.WebSocketClientProtocol, shutdown_event: asyncio.Event):
    while not shutdown_event.is_set():
        try:
            await ws.send(json.dumps({"method": "ping"}))
        except Exception:
            return
        await asyncio.sleep(PING_EVERY)

async def monitor_raw_messages():
    """Connect to WebSocket and print raw messages"""
    if not LEADER_ADDRESS or LEADER_ADDRESS == "0x...":
        print("❌ Please set LEADER_ADDRESS in the script")
        return

    print(f"Connecting to {WS_URL}")

    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        del signum, frame
        print("\nShutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Simple reconnect loop
    backoff = 1

    while not shutdown_event.is_set():
        try:
            async with websockets.connect(WS_URL) as websocket:
                print(f"✅ WebSocket connected!")

                # Subscribe to user events (positions, fills, TP/SL updates) and orders
                subscriptions = [
                    {
                        "method": "subscribe",
                        "subscription": {"type": "userEvents", "user": LEADER_ADDRESS},
                    },
                    {
                        "method": "subscribe",
                        "subscription": {"type": "orderUpdates", "user": LEADER_ADDRESS},
                    },
                    # {"method": "subscribe", "subscription": {"type": "userFills", "user": LEADER_ADDRESS, "aggregateByTime": True}},
                ]

                for sub in subscriptions:
                    await websocket.send(json.dumps(sub))

                heartbeat_task = asyncio.create_task(hl_heartbeat(websocket, shutdown_event))

                print(f"Monitoring ALL user events and orders: {LEADER_ADDRESS}")
                print("=" * 80)

                
                async for message in websocket:
                    if shutdown_event.is_set():
                        break
                    
                    try:
                            
                        data = json.loads(message)

                    except json.JSONDecodeError:
                        print("⚠️ Received invalid JSON")
                    except Exception as e:
                        print(f"❌ Error: {e}")

                    # Ignore ping/pong messages
                    if data.get("channel") == "ping" or data.get("channel") == "pong":
                        continue

                    print(f"RAW MESSAGE: {json.dumps(data, indent=2)}")
                    print("-" * 40)
                
            # Reset backoff after successful connection
            backoff = 1

        except websockets.exceptions.ConnectionClosed as e:
            print(f"WebSocket closed: code={getattr(e,'code',None)} reason={getattr(e,'reason',None)}")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
        
        if not shutdown_event.is_set():
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30) # Exponential backoff up to 30 seconds


async def main():
    print("Raw WebSocket Message Monitor")
    print("=" * 40)

    if not WS_URL:
        print("❌ Missing HYPERLIQUID_TESTNET_PUBLIC_WS_URL in .env file")
        return

    await monitor_raw_messages()


if __name__ == "__main__":
    asyncio.run(main())
