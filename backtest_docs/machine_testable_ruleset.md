# Machine-Testable Trading Ruleset

This document converts our notes into explicit, testable rules. It is designed to be implemented by code with no discretionary interpretation.

## 1) Scope and Assumptions

- **Instruments**: Index futures or equities with liquid intraday data (e.g., NQ/ES, SPY, QQQ).
- **Timezone**: America/New_York (ET).
- **Required data**:
  - OHLCV at 1m, 5m, 15m, 1h, 4h.
  - (Optional) Correlated asset for SMT: ES vs NQ.
- **Tick size**: instrument-specific (e.g., NQ = 0.25).

## 2) Parameter Defaults

- **True Day Open**: 09:30:00 ET (NY cash open).
- **Session Times**:
  - Asia: 20:00–03:00 ET
  - London: 03:00–08:00 ET
  - NY: 08:00–16:00 ET
- **Stop Loss (base)**: 50 ticks (fixed) from entry.
- **Minimum sweep distance**: 1 tick beyond swing high/low.
- **Execution timeframes**:
  - If 1H trend == 4H trend: use 5m
  - If 1H trend != 4H trend: use 15m

## 3) Definitions (Strict)

### 3.1 Swing High / Swing Low (3-candle rule)
- **Swing High**: candle `i` has a higher high than `i-1` and `i+1`.
- **Swing Low**: candle `i` has a lower low than `i-1` and `i+1`.

### 3.2 Trend (per timeframe)
- **Uptrend**: most recent swing high > previous swing high AND most recent swing low > previous swing low.
- **Downtrend**: most recent swing high < previous swing high AND most recent swing low < previous swing low.
- **No trend**: otherwise.

### 3.3 Break of Structure (BOS)
- **Bullish BOS**: candle body closes above the most recent swing high.
- **Bearish BOS**: candle body closes below the most recent swing low.

### 3.4 Fair Value Gap (FVG)
Given three consecutive candles `c1`, `c2`, `c3`:
- **Bullish FVG** exists if `c1.high < c3.low`.
  - Gap range = `[c1.high, c3.low]`.
- **Bearish FVG** exists if `c1.low > c3.high`.
  - Gap range = `[c3.high, c1.low]`.
- **FVG is “tapped”** when price enters the gap range.

### 3.5 Liquidity Pool and Sweep
- **Liquidity Pool**: price at any recent swing high/low.
- **Liquidity Sweep**:
  - Price trades at least 1 tick beyond a swing high/low,
  - Then closes back inside the prior range within the next 3 candles.

### 3.6 Equilibrium (Premium / Discount)
- **Equilibrium**: midpoint between last major swing high and swing low on the **bias timeframe** (4h).
- **Discount**: price < midpoint (prefer longs).
- **Premium**: price > midpoint (prefer shorts).

### 3.7 Order Block (OB)
- **Bullish OB**: last down candle before a bullish BOS; OB range = candle body.
- **Bearish OB**: last up candle before a bearish BOS; OB range = candle body.

### 3.8 Breaker Block (BB)
- A prior OB that failed and flipped:
  - **Bullish BB**: bearish OB is broken to the upside and later retested from above.
  - **Bearish BB**: bullish OB is broken to the downside and later retested from below.

### 3.9 Balanced Price Range (BPR)
- Overlap between a bullish FVG and a bearish FVG within the last 20 candles.
- Overlap zone used as a reversal confluence.

### 3.10 Session Highs / Lows
- **Asia High/Low**: high/low of Asia session.
- **London High/Low**: high/low of London session.
- **NY High/Low**: high/low of NY session (optional target).
- Use the most recent completed session.

### 3.11 True Day Open
- Price at 09:30:00 ET.

## 4) Confidence Interval Strategy (Position Sizing + Management)

### 4.1 Confidence Score (0–10)
Score each setup using the checklist below (sum points):
- **Trend alignment (4H + 1H)**: +2
- **Liquidity sweep at HTF level**: +2
- **BOS on execution timeframe**: +2
- **Entry at HTF FVG/OB/BB**: +1
- **Entry in discount (longs) / premium (shorts)**: +1
- **Session confluence (Asia/London High/Low or True Day Open)**: +1
- **SMT divergence (ES vs NQ)**: +1

### 4.2 Tier Mapping
- **Low (1–3)**: score 0–3
- **Medium (4–6)**: score 4–6
- **High (7–10)**: score 7+

### 4.3 Position Size (Futures, example)
- **Low**: 1–2 micros
- **Medium**: 3–6 micros
- **High**: 7–10 micros (or 1–2 minis)

### 4.4 Risk & Trade Management (All trades)
- **Stop Loss**: fixed 50 ticks from entry.
- **Break-even rules**:
  - Low: move SL to entry at +50 ticks.
  - Medium: move SL to entry at +75 ticks.
  - High: move SL to entry at +100 ticks.
- **Take Profit**:
  - Low: full exit at 2R.
  - Medium: full exit at 3R.
  - High: partial exit at 4R (close 30–50%), move SL to +1R, runner targets next HTF liquidity or session level.

## 5) Systemized Entry Rules (Machine-Testable)

### 5.1 Determine Bias
1. Compute 4H trend.
2. Compute 1H trend.
3. If 1H matches 4H → execution timeframe = 5m.
4. If 1H opposes 4H → execution timeframe = 15m.

### 5.2 Precondition (Wait for HTF interaction)
Entry is only allowed after **one** of the following on 1H or 4H:
- Liquidity sweep of an HTF swing high/low, OR
- Price taps an HTF FVG/OB/BB.

### 5.3 Trigger (Lower Timeframe)
After precondition:
1. Wait for BOS on execution timeframe in the **direction of intended trade**.
2. Identify the first valid confluence zone created by the BOS:
   - LTF FVG, or LTF OB, or LTF BB, or Equilibrium.
3. **Entry**: first retest of that confluence zone within the next 10 candles.

### 5.4 Direction Rules
- **Longs**:
  - Prefer when 4H trend is bullish.
  - Entry zone should be in discount (below 50%).
- **Shorts**:
  - Prefer when 4H trend is bearish.
  - Entry zone should be in premium (above 50%).

## 6) Targets (Priority Order)

1. Opposing session high/low (Asia/London).
2. Opposing HTF liquidity (previous swing high/low).
3. Opposing HTF FVG/OB/BB.
4. Prior day high/low (if available).

## 7) Backtest Output Requirements

- Win rate, average R, max drawdown.
- Separate stats by confidence tier.
- Separate stats by:
  - Trend alignment (1H/4H match vs mismatch).
  - Session (London/NY).
  - Instrument (NQ/ES/SPY).

## 8) Notes

- This ruleset intentionally removes discretionary language (“looks like,” “should,” “manipulation”).
- Any missing feature (e.g., SMT) can be toggled off by removing its score contribution.
- If no setup meets the precondition + trigger rules, **no trade** is taken.
