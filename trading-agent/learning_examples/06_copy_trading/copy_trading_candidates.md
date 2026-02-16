Absolutely — below is a **clean, copy-paste-ready Markdown version** of the explanation, with **clear sections, tables, and LaTeX formulas**. You can drop this straight into a README, research note, or Notion doc.

---

# Scoring Parameter Rationale (Copy-Trading Leader Selection)

This document explains the **default parameter values** used in the longevity-weighted trader scoring model and the **reasoning behind each choice**. These parameters encode assumptions about **risk, skill, replicability, and survivorship** when selecting traders to copy.

---

## 1. `MIN_MONTH_VOLUME = 10_000_000`

### Purpose

Filters for **sufficient trading activity** so that a trader’s performance is:

* statistically meaningful
* replicable via copy trading
* not driven by a small number of lucky trades

### Definition

A trader must have at least:

[
\text{30D Trading Volume} \ge $10{,}000{,}000
]

### Why This Value

* On Hyperliquid, $10M/month implies **regular participation**
* Below this threshold, many traders:

  * trade sporadically
  * show unstable strategies
  * have PnL dominated by single trades
* This level still includes **non-whale** traders

### Trade-offs

| Too Low                     | Too High          |
| --------------------------- | ----------------- |
| Noisy, low-activity traders | Whale-only bias   |
| Lucky one-offs              | Reduced diversity |

### When to Adjust

| Goal                      | Change   |
| ------------------------- | -------- |
| Include swing traders     | ↓ 3–5M   |
| Active scalping           | ↑ 25–50M |
| Conservative copy trading | ↑ 50M+   |

---

## 2. `MIN_ACCOUNT_VALUE = 100_000`

### Purpose

Removes **statistically unstable small accounts** whose returns are dominated by leverage or variance.

### Definition

[
\text{Account Value} \ge $100{,}000
]

### Why This Value

* Large enough to:

  * survive drawdowns
  * trade multiple markets
* Small enough to:

  * include skilled non-whales
* Below $50K, ROI is often **variance-driven**

### Trade-offs

| Too Low                   | Too High                |
| ------------------------- | ----------------------- |
| Leverage lottery winners  | Miss emerging talent    |
| Unrealistic risk profiles | Over-institutional bias |

### When to Adjust

| Scenario               | Change    |
| ---------------------- | --------- |
| Small copy size        | ↓ 25–50K  |
| Institutional profiles | ↑ 250K–1M |
| Capital preservation   | ↑ 500K    |

---

## 3. `ROI_NORMALIZER = 0.25` (25%)

### Purpose

Prevents **extreme ROI values** from dominating the score while still rewarding strong returns.

### Formula

ROI is transformed using:

[
\text{ROI}_{scaled} = \tanh\left(\frac{\text{ROI}}{0.25}\right)
]

### Why This Value

* 25% ROI over a window is already **excellent**
* Saturation behavior:

  * 25% → ( \tanh(1) \approx 0.76 )
  * 50% → ( \tanh(2) \approx 0.96 )
  * 100% → ( \tanh(4) \approx 0.999 )
* Prevents leverage and moonshots from appearing as “skill”

### Trade-offs

| Too Small               | Too Large        |
| ----------------------- | ---------------- |
| Flattens strong traders | Rewards leverage |
| Low differentiation     | Chases tail risk |

### When to Adjust

| Risk Style   | Change      |
| ------------ | ----------- |
| Conservative | ↓ 0.15      |
| Balanced     | 0.25        |
| Aggressive   | ↑ 0.40–0.75 |

---

## 4. `DISPERSION_PENALTY_K = 3.0`

### Purpose

Penalizes **inconsistent performance across time horizons**.

### Formula

Let ( \sigma_{ROI} ) be the standard deviation of ROI across windows:

[
M_{smooth} = \frac{1}{1 + k \cdot \sigma_{ROI}}
]

### Why This Matters

A trader with wildly different ROIs across:

* day
* week
* month
* all-time

…is not predictable or reliably copyable.

### Why This Value

* Mild volatility → small penalty
* Spiky equity curves → heavily penalized
* Still allows volatility (this is crypto)

### Trade-offs

| Too Low            | Too High           |
| ------------------ | ------------------ |
| Chases spikes      | Over-smooth bias   |
| High drawdown risk | Suppresses returns |

### When to Adjust

| Preference        | Change |
| ----------------- | ------ |
| Stable income     | ↑ 4–6  |
| Momentum trading  | ↓ 1–2  |
| Conservative copy | ↑ 6+   |

---

## 5. `TARGET_MONTH_VOLUME = 50_000_000`

### Purpose

Provides a **soft activity threshold** so volume helps only until it’s “enough”.

### Formula

[
M_{volume} = \sigma\left(\log(1 + V_{month}) - \log(1 + V_{target})\right)
]

### Why This Exists

* Ensures traders are active
* Prevents “highest volume wins” bias
* Saturates impact beyond sufficiency

### Why This Value

* $50M/month ≈ serious, consistent trader
* Above this level, extra volume ≠ extra skill
* Keeps whales from dominating by size alone

### Trade-offs

| Too Low               | Too High          |
| --------------------- | ----------------- |
| Inactive traders pass | Whale dominance   |
| Volume irrelevant     | Reduced diversity |

### When to Adjust

| Use Case            | Change     |
| ------------------- | ---------- |
| Small copy size     | ↓ 20–30M   |
| Institutional style | ↑ 100–200M |
| HFT bias            | ↑ 250M     |

---

## 6. Parameter Interaction (Important)

These parameters **must be tuned together**:

| Parameter              | Role                 |
| ---------------------- | -------------------- |
| `MIN_*`                | Eligibility          |
| `ROI_NORMALIZER`       | Return saturation    |
| `DISPERSION_PENALTY_K` | Stability            |
| `TARGET_MONTH_VOLUME`  | Activity sufficiency |

### Example Couplings

* ↑ `ROI_NORMALIZER` → ↑ `DISPERSION_PENALTY_K`
* ↓ `MIN_MONTH_VOLUME` → ↑ consistency penalties

---

## Summary

These defaults are designed to:

* favor **long-term skill over short-term luck**
* reduce tail-risk and leverage bias
* surface **copy-trade-able** traders

They are **strong starting points**, not universal truths.

---

If you want next:

* sensitivity analysis across parameters
* automated tuning via historical snapshots
* multiple trader personas (safe / balanced / aggressive)

Just say the word.
