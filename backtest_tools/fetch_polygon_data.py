import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def fetch_aggregates(symbol: str, start: str, end: str, multiplier: int, timespan: str, api_key: str) -> dict:
    """Fetch aggregate bars from Polygon.io API."""
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    url = f"{base}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "stonks-backtest"})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 429:
            print(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            with urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        raise


def fetch_all_pages(symbol: str, start: str, end: str, multiplier: int, timespan: str, api_key: str) -> list[dict]:
    """Fetch all pages of data (handles pagination for large date ranges)."""
    all_results = []
    current_start = start

    while True:
        payload = fetch_aggregates(symbol, current_start, end, multiplier, timespan, api_key)
        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            print(f"Polygon error for {symbol}: {payload.get('error', payload)}")
            break
        if status == "DELAYED" and not all_results:
            print(f"Polygon data for {symbol} is delayed (free tier).")

        results = payload.get("results", [])
        if not results:
            break

        all_results.extend(results)
        print(f"  Fetched {len(results)} bars (total: {len(all_results)})")

        # Check if there's more data (pagination)
        if payload.get("next_url"):
            # Get timestamp of last bar and continue from there
            last_ts = results[-1]["t"]
            # Convert ms to date string
            last_date = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            current_start = last_date.strftime("%Y-%m-%d")
            time.sleep(0.5)  # Small delay between pages
        else:
            break

    return all_results


def write_csv(symbol: str, results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for item in results:
            writer.writerow(
                [
                    item["t"],
                    item["o"],
                    item["h"],
                    item["l"],
                    item["c"],
                    item.get("v", 0),
                ]
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Fetch Polygon aggregates to CSV.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., AAPL,MSFT,SPY)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--multiplier", type=int, default=1, help="Aggregate multiplier (default 1)")
    parser.add_argument("--timespan", default="minute", help="Timespan (minute, hour, day)")
    parser.add_argument("--output", default="data/polygon", help="Output directory")
    parser.add_argument("--delay", type=int, default=15, help="Delay between symbols in seconds (default 15 for rate limiting)")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY") 
    if not api_key or api_key == "redacted":
        raise SystemExit("POLYGON_API_KEY is not set. Set it via environment variable.")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_dir = Path(args.output)

    for i, symbol in enumerate(symbols):
        print(f"\nFetching {symbol} ({i+1}/{len(symbols)})...")

        results = fetch_all_pages(symbol, args.start, args.end, args.multiplier, args.timespan, api_key)

        if not results:
            print(f"No data for {symbol} in range.")
            continue

        output_path = output_dir / f"{symbol}_{args.start}_{args.end}_{args.multiplier}{args.timespan}.csv"
        write_csv(symbol, results, output_path)
        print(f"Wrote {output_path} ({len(results)} bars)")

        # Rate limit delay between symbols (skip on last symbol)
        if i < len(symbols) - 1:
            print(f"Waiting {args.delay}s for rate limit...")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()


# Use: python fetch_polygon_data.py --symbols AAPL,MSFT,SPY --start 2024-01-01 --end 2024-12-31 --timespan minute --output ../data/polygon
# to fetch data for the year 2024 for the symbols AAPL, MSFT, and SPY.