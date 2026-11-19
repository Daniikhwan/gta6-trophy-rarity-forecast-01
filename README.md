[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Daniikhwan/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Daniikhwan/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

`main.py` forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the percent of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. It reads a case file from the input directory and writes one forecast file per case to the output directory.

The model is an ensemble of three components in logit space: a counts model that forecasts tracked players and each trophy's earners separately (weight 0.35), a trend model that fits each trophy's logit earn rate against log time over the last 24 hours (weight 0.55), and persistence on the last observed rate (weight 0.10). A ceiling prior from each trophy's tier and description caps forecasts at the trophy's plausible mature rarity, and a refresh-dilution haircut handles stale histories: Exophase refreshes its counts hours apart, so a flat page that is hours old is about to be diluted by newly tracked players holding fewer trophies. The spread combines a random-walk term, the disagreement between components, and a floor, so the quantiles stay honest about a refresh the program has not seen.

Validation on the two practice cases (checkpoints 03:00Z and 03:30Z, 19 Nov 2026, scored against the refreshed 04:30Z page): mean pinball loss on the logit scale **0.246**, versus **1.51** for the persistence baseline; typical miss 0.10 percentage points, 90% coverage 1.00. Runs are deterministic (byte-identical outputs) and take well under a second per case.

## Run it

```sh
python3 -P -s main.py
```

Set `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` to point at the case files and the output directory (defaults `/input` and `/output`). Pure Python 3.13 standard library, no network, no wall clock, no randomness.

## Method

Data source: the house's recording of Exophase's GTA VI trophy page (https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/), as shipped in each case file's `history`. Hours are measured since 2026-11-19T00:00:00Z. History is first collapsed to distinct reads (a read counts only when its players-tracked count or rates differ from the previous kept read), so flat stretches between Exophase refreshes carry no weight. The counts model fits ln(tracked players) on ln(1+h) and ln(earners) on ln(1+h) per trophy, averaging the full-history and last-24h slopes and damping the earner slope toward the tracked slope over the horizon. The curve model fits logit(rate) on ln(1+h) over the last 24 hours and reads the line at the target. The ensemble median is the weighted mean of the three components' logits; quantiles use z = (-1.645, -0.674, 0, 0.674, 1.645) with sigma from the horizon random walk, component disagreement and a 0.15 floor, clamped to [0.35, 2.0]. When the history has at most 2 distinct reads and at least 3 hours have elapsed since the first read, the median is multiplied by 0.30 and 0.50² is added inside the sigma square root (refresh-dilution haircut). A trophy whose last rate is >= 99.99% forecasts [99, 100, 100, 100, 100]. Every quantile is clamped to the trophy's ceiling prior and rounded half-up to 4 decimals; any unexpected per-trophy failure falls back to the last rate with the starter-style spread so a case never crashes.

<sub>Entered on [make.fun](https://make.fun/?ref=vnsny7za&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
