
# 📊 Weekly Scanner Comparison: `163026` → `132933`

---

## 🔍 Timeframe Comparison
| Field                  | Previous | Current | Change |
|------------------------|----------|---------|--------|
| **Avg RSI**            | 54.26 | 70.02 | ⬆️ |
| **Top Composite Score**| 98.46 | 109.16 | ⬆️ |
| **Overbought Count**   | 1 | 10 | ⬆️ |
| **Avg Momentum Short** | 0.13 | 0.23 | ⬆️ |
| **Avg Volume Ratio**   | 2.54 | 2.13 | ⬇️ |
| **Volatility (StdDev)**| 10.1 | 10.75 | ⬆️ |
| **Max Drawdown (%)**   | -45.85 | -44.16 | ⬇️ |

---

## 🏷️ Signal Distribution
### Previous
Neutral: 13, Weak Buy: 11, Buy: 4, Strong Buy: 1, Sell: 1

### Current
Buy: 27, Weak Buy: 3

---

## 🔄 Turnover Rate
**80.0%** of the current top 30 are new this week.

---

## 🚀 New Leaders (Not in Top 30 last week)
- `1000000MOG/USDT:USDT`
- `ADA/USDT:USDT`
- `ALGO/USDT:USDT`
- `ALPINE/USDT:USDT`
- `ARKM/USDT:USDT`
- `CATI/USDT:USDT`
- `CRO/USDT:USDT`
- `DASH/USDT:USDT`
- `ENA/USDT:USDT`
- `ENS/USDT:USDT`
- `ETH/USDT:USDT`
- `GRT/USDT:USDT`
- `HBAR/USDT:USDT`
- `KAVA/USDT:USDT`
- `LTC/USDT:USDT`
- `NEO/USDT:USDT`
- `OMNI/USDT:USDT`
- `OP/USDT:USDT`
- `QTUM/USDT:USDT`
- `RENDER/USDT:USDT`
- `STRK/USDT:USDT`
- `WAVES/USDT:USDT`
- `XRP/USD:XRP`
- `XRP/USDT:USDT`

## 📉 Fallen from Top 30
- `AAVE/USDT:USDT`
- `ARB/USDT:USDT`
- `ASR/USDT:USDT`
- `ATOM/USDT:USDT`
- `AXS/USDT:USDT`
- `BCH/USDT:USDT`
- `BSV/USDT:USDT`
- `BTC/USD:BTC`
- `BTC/USDT:USDT`
- `ETC/USDT:USDT`
- `FTT/USDT:USDT`
- `ILV/USDT:USDT`
- `MANA/USDT:USDT`
- `MKR/USDT:USDT`
- `ORCA/USDT:USDT`
- `PAXG/USDT:USDT`
- `QNT/USDT:USDT`
- `SAND/USDT:USDT`
- `SONIC/USDT:USDT`
- `SOON/USDT:USDT`
- `TIA/USDT:USDT`
- `UNI/USDT:USDT`
- `XMR/USDT:USDT`
- `ZEN/USDT:USDT`

## 🔄 Consistent Performers
- `BIO/USDT:USDT`
- `BNB/USDT:USDT`
- `ETH/USD:ETH`
- `GAS/USDT:USDT`
- `KCS/USDT:USDT`
- `STX/USDT:USDT`

---

## 📈 Biggest Movers (Top 5 Up/Down by Composite Score)
**Up:**
- `ETH/USD:ETH` (+29.51)
- `GAS/USDT:USDT` (+27.66)
- `KCS/USDT:USDT` (+24.56)
- `STX/USDT:USDT` (+18.7)
- `BNB/USDT:USDT` (+15.31)

**Down:**
- `BIO/USDT:USDT` (11.45)
- `BNB/USDT:USDT` (15.31)
- `STX/USDT:USDT` (18.7)
- `KCS/USDT:USDT` (24.56)
- `GAS/USDT:USDT` (27.66)

---

## 💹 Top 5 Price Gainers/Losers (Common Symbols)
**Gainers:**
- `ETH/USD:ETH` (+14.58%)
- `KCS/USDT:USDT` (+11.84%)
- `GAS/USDT:USDT` (+10.58%)
- `STX/USDT:USDT` (+7.48%)
- `BNB/USDT:USDT` (+3.95%)

**Losers:**
- `KCS/USDT:USDT` (11.84%)
- `GAS/USDT:USDT` (10.58%)
- `STX/USDT:USDT` (7.48%)
- `BNB/USDT:USDT` (3.95%)
- `BIO/USDT:USDT` (0.0%)


---

## ⚠️ Risk Warnings
| Warning Type         | Prev | Curr | Change |
|----------------------|------|------|--------|
| **RSI > 75**         | {prev_ob} | {curr_ob} | {'⬆️' if curr_ob > prev_ob else '⬇️'} |
| **Momentum > 10%**   | {count_gt(prev, 'momentum_short', 0.1)} | {count_gt(curr, 'momentum_short', 0.1)} | {'⬆️' if count_gt(curr, 'momentum_short', 0.1) > count_gt(prev, 'momentum_short', 0.1) else '⬇️'} |

---

## ✅ Top 3 Risk-Adjusted Picks (RSI < 70, Momentum > 5%)
{chr(10).join([f"- `{row['symbol']}` (RSI: {row['rsi']}, Momentum: {round(row['momentum_short']*100,1)}%)" 
              for _, row in curr[(curr["rsi"] < 70) & (curr["momentum_short"] > 0.05)]
              .sort_values("composite_score", ascending=False)
              .head(3)
              .iterrows()])}

---

## 🏅 Performance of Last Week's Top 5 in Current Scan
{chr(10).join([f"- `{sym}`: Composite Score: {lastweek_perf.loc[sym]['composite_score'] if sym in lastweek_perf.index else 'N/A'}" for sym in lastweek_top])}

---

## 📊 Summary Statistics (Composite Score)
| Metric | Previous | Current |
|--------|----------|---------|
| {summary_prev} | {summary_curr} |

---

> Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
