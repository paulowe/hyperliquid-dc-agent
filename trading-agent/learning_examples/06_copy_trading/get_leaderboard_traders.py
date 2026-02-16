import httpx
# Ordered by 30D volume by default
URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

with httpx.Client(timeout=60) as client:
    r = client.get(URL)
    r.raise_for_status()
    data = r.json()

print(f"Entries: {len(data)}")
print(data[0].keys())
