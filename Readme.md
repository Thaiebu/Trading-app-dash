# Quick Reference Card - Advanced Options Dashboard

## 🎯 3 Sections at a Glance

### 📊 Section 1: Call vs Put OI
**What:** Current OI distribution
**Chart:** Grouped bars (Green=Call, Red=Put)
**Key Info:** PCR, Max OI strikes
**Use For:** Support/resistance identification

### 📈 Section 2: OI Change (Bar)
**What:** OI change from last interval
**Chart:** Bars showing changes
**Key Info:** Where positions added/removed
**Use For:** Accumulation/distribution zones

### 📉 Section 3: OI Change (Line)
**What:** OI trends over time
**Chart:** 3 lines (CE, PE, Futures)
**Key Info:** Directional bias over time
**Use For:** Trend confirmation

---

## 🎛️ Controls Quick Guide

| Control | Options | Purpose |
|---------|---------|---------|
| Symbol | NIFTY50, BANKNIFTY, FINNIFTY | Select index |
| Expiry | Next 4 Thursdays | Choose expiry |
| Interval | 1/3/5/15Min, 1H | Time granularity |
| Range | Checkbox + Start/End | Filter strikes |
| Mode | Live / Historical | Data source |

---

## 📊 PCR Interpretation

| PCR Value | Sentiment | Action |
|-----------|-----------|--------|
| < 0.7 | Extremely Bullish | Caution (overbought) |
| 0.7 - 0.9 | Bullish | Upside likely |
| 0.9 - 1.1 | Neutral | Range-bound |
| 1.1 - 1.3 | Bearish | Downside risk |
| > 1.3 | Extremely Bearish | Potential reversal |

---

## 🎯 Trading Signals

### Bullish Signals
✓ PCR < 0.8
✓ Call OI buildup at higher strikes
✓ Rising Call OI + Rising Futures OI
✓ Put unwinding at support

### Bearish Signals
✓ PCR > 1.3
✓ Put OI buildup at lower strikes
✓ Rising Put OI + Falling Futures OI
✓ Call unwinding at resistance

### Neutral Signals
✓ PCR ~1.0
✓ Balanced OI distribution
✓ Minimal OI changes
✓ Flat trends in Section 3

---

## 🚀 Quick Start

```bash
# Install
pip install dash plotly pandas numpy

# Run
python advanced_live_dashboard.py

# Access
http://127.0.0.1:8050
```

---

## 🔍 Analysis Workflow

1. **Set Parameters** → Symbol, Expiry, Interval
2. **Check Section 1** → PCR, Max OI strikes
3. **Check Section 2** → OI changes, buildup zones
4. **Check Section 3** → OI trends, futures divergence
5. **Make Decision** → Combine all insights

---

## ⚡ Keyboard Shortcuts (Browser)

- `Ctrl + R` → Refresh dashboard
- `Ctrl + +` → Zoom in charts
- `Ctrl + -` → Zoom out charts
- `Ctrl + 0` → Reset zoom

---

## 📊 Chart Interactions

- **Hover** → See exact values
- **Click Legend** → Toggle series
- **Double-Click Legend** → Isolate series
- **Drag** → Zoom to selection
- **Double-Click Chart** → Reset zoom

---

## 🎓 Pro Tips

1. Enable Range Filter for ATM analysis
2. Use 3Min interval for intraday
3. Use 15Min/1H for swing trading
4. Compare current OI with historical
5. Watch futures OI for confirmation
6. Focus on high volume + high OI strikes

---

Print this card for quick reference during trading!
