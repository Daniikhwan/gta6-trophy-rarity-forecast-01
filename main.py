"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target as five quantiles (5/25/50/75/95%).

Model (pure Python 3.13 stdlib, deterministic, no network, no wall clock):
  - Collapse history to distinct reads (a read counts only when its
    (players_tracked, rates) differs from the previous kept read).
  - Component A (weight 0.35): counts model. Fit ln(tracked) on ln(1+h);
    fit ln(earners) on ln(1+h) per trophy (full history and last-24h slopes
    averaged, damped toward the tracked slope over the horizon); rate =
    100 * earners(target) / tracked(target).
  - Component B (weight 0.55): curve model. Least-squares line of
    logit(rate) on ln(1+h) over the last 24h of distinct reads, read at
    the target hour.
  - Component C (weight 0.10): persistence, the trophy's last rate.
  - Ceiling prior (mature-rarity anchor from tier + description) caps
    components A and B.
  - Ensemble median is the weighted mean of the three components' logits;
    spread combines a random-walk term, component disagreement and a floor.
  - Flat-history refresh-dilution haircut: with at most 2 distinct reads
    and >= 3h since the first read, Exophase is stale and a refresh is due
    before the target, so the median is multiplied by 0.30 and the spread
    widened (a refresh dilutes every rate as newly tracked players join).

Hours h are measured since 2026-11-19T00:00:00Z. All times come from the
case file. Quantiles are rounded half-up to 4 decimals (JavaScript
Math.round style); whole numbers are written as integers.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def parse_time(text):
    """ISO 8601 timestamp -> datetime (UTC). Returns None if unparseable."""
    if not text or not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_epoch(dt):
    if dt is None:
        return 0.0
    return (dt - EPOCH).total_seconds() / 3600.0


def clamp_pct(p):
    return min(99.99, max(0.01, p))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, JavaScript Math.round style."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def least_squares(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def ceiling_prior(trophy):
    """Mature-rarity anchor: a trophy's week-one rate cannot exceed this."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return 5.0
    if "100% completion" in desc or "attain 100%" in desc:
        return 6.0
    if "chapter 1" in desc:
        return 85.0
    if "chapter 2" in desc:
        return 65.0
    if "chapter 3" in desc:
        return 58.0
    if "chapter 4" in desc:
        return 55.0
    if "final mission" in desc:
        return 50.0
    if ("250 stunts" in desc or "50 challenges" in desc
            or "collectible card sets" in desc or "waink it at 100" in desc):
        return 10.0
    if "gold medal" in desc or "top fun drone race at every" in desc:
        return 12.0
    if tier == "gold":
        return 30.0
    if tier == "silver":
        return 45.0
    return 40.0


def distinct_reads(history):
    """Collapse history to distinct reads, keeping order (oldest first)."""
    kept = []
    prev_key = None
    for read in history or []:
        rates = read.get("rates")
        key = (read.get("players_tracked"),
               tuple(rates) if isinstance(rates, list) else None)
        if key != prev_key:
            kept.append(read)
            prev_key = key
    return kept


def last_rate(reads, idx):
    """Last non-null rate for trophy idx across reads, else None."""
    for read in reversed(reads):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def forecast_trophy(idx, trophy, reads, h_at, h_target, horizon_days,
                     n_slope, n_target, haircut):
    ceiling = ceiling_prior(trophy)
    last = last_rate(reads, idx)
    last_r = last if last is not None else 0.01

    # Trophy at (essentially) 100%: every newly tracked player holds it.
    if last is not None and last >= 99.99:
        return [99, 100, 100, 100, 100]

    # Hours of each distinct read, from its slot (fall back to its `at`).
    hours = []
    for read in reads:
        dt = parse_time(read.get("slot")) or parse_time(read.get("at"))
        hours.append(hours_since_epoch(dt))
    h_last = hours[-1] if hours else h_at

    # --- Component A: counts model -------------------------------------
    rate_a = None
    if reads:
        # Earners per distinct read for this trophy.
        points = []  # (h, E) with E > 0
        e_last = None
        for read, h in zip(reads, hours):
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            n = read.get("players_tracked")
            if not isinstance(n, (int, float)) or n <= 0:
                continue
            e = rates[idx] * n / 100.0
            if e > 0:
                points.append((h, e))
            e_last = e
        if e_last is not None and e_last <= 0:
            rate_a = last_r
        elif points:
            slopes = []
            xs_all = [math.log1p(h) for h, _ in points]
            ys_all = [math.log(e) for _, e in points]
            if len(points) >= 3:
                fit = least_squares(xs_all, ys_all)
                if fit is not None:
                    slopes.append(fit[1])
            recent = [(h, e) for h, e in points if h >= h_at - 24.0]
            if len(recent) >= 3:
                fit = least_squares([math.log1p(h) for h, _ in recent],
                                    [math.log(e) for _, e in recent])
                if fit is not None:
                    slopes.append(fit[1])
            b = sum(slopes) / len(slopes) if slopes else 0.0
            b = min(2.5, max(0.0, b))
            damp = min(0.6, max(0.0, 1.0 - math.sqrt((1.0 + h_last) / (1.0 + h_target))))
            b = (1.0 - damp) * b + damp * n_slope
            e_target = points[-1][1] * ((1.0 + h_target) / (1.0 + h_last)) ** b
            if n_target and n_target > 0:
                rate_a = 100.0 * e_target / n_target
    if rate_a is None:
        rate_a = last_r
    rate_a = min(ceiling, max(0.01, rate_a))

    # --- Component B: curve model --------------------------------------
    rate_b = None
    recent_pts = []
    for read, h in zip(reads, hours):
        if h < h_at - 24.0:
            continue
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            recent_pts.append((h, logit(rates[idx])))
    if len(recent_pts) >= 3:
        fit = least_squares([math.log1p(h) for h, _ in recent_pts],
                            [y for _, y in recent_pts])
        if fit is not None:
            a, b = fit
            rate_b = inv_logit(a + b * math.log1p(h_target))
    if rate_b is None:
        rate_b = last_r
    rate_b = min(ceiling, max(0.01, rate_b))

    # --- Component C: persistence --------------------------------------
    rate_c = last_r

    # --- Ensemble -------------------------------------------------------
    lg_a, lg_b, lg_c = logit(rate_a), logit(rate_b), logit(rate_c)
    med_logit = 0.35 * lg_a + 0.55 * lg_b + 0.10 * lg_c
    mean_lg = (lg_a + lg_b + lg_c) / 3.0
    disagreement = math.sqrt(
        ((lg_a - mean_lg) ** 2 + (lg_b - mean_lg) ** 2
         + (lg_c - mean_lg) ** 2) / 3.0)
    sigma2 = ((0.5 * math.sqrt(horizon_days)) ** 2
              + (0.3 * disagreement) ** 2 + 0.15 ** 2)
    median = inv_logit(med_logit)
    if haircut:
        median = median * 0.30
        sigma2 += 0.50 ** 2
    sigma = min(2.0, max(0.35, math.sqrt(sigma2)))
    med_lg = logit(median)
    qs = [round4(inv_logit(med_lg + z * sigma)) for z in Z]
    # The ceiling prior is a hard mature-rarity anchor: no quantile may
    # exceed it (the spread alone must not push q95 past what the trophy
    # can ever reach). Clamp, then re-enforce non-decreasing order.
    qs = [min(q, ceiling) for q in qs]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case, name):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    horizon_days = horizon_minutes / 1440.0
    at_dt = parse_time(case.get("at"))
    h_at = hours_since_epoch(at_dt)
    target_dt = parse_time(case.get("target"))
    h_target = hours_since_epoch(target_dt) if target_dt is not None else h_at + horizon_minutes / 60.0
    if h_target <= h_at:
        h_target = h_at + horizon_minutes / 60.0

    reads = distinct_reads(history)

    # Tracked-player count model: ln(N) on ln(1+h), slope clamped [0, 2].
    n_slope = 0.0
    n_target = None
    if reads:
        n_pts = []
        for read in reads:
            n = read.get("players_tracked")
            if isinstance(n, (int, float)) and n > 0:
                dt = parse_time(read.get("slot")) or parse_time(read.get("at"))
                n_pts.append((hours_since_epoch(dt), n))
        if len(n_pts) >= 2:
            fit = least_squares([math.log1p(h) for h, _ in n_pts],
                                [math.log(n) for _, n in n_pts])
            if fit is not None:
                a, b = fit
                n_slope = min(2.0, max(0.0, b))
                n_target = math.exp(a + n_slope * math.log1p(h_target))
        last_n = n_pts[-1][1] if n_pts else None
        if n_target is None:
            n_target = last_n
        elif last_n is not None:
            n_target = max(n_target, last_n)

    # Flat-history refresh-dilution haircut (see module docstring).
    haircut = False
    if len(reads) <= 2 and reads:
        first_dt = parse_time(reads[0].get("slot")) or parse_time(reads[0].get("at"))
        if first_dt is not None and at_dt is not None:
            if (at_dt - first_dt).total_seconds() / 3600.0 >= 3.0:
                haircut = True

    rows = []
    for idx, trophy in enumerate(trophies):
        try:
            qs = forecast_trophy(idx, trophy, reads, h_at, h_target,
                                 horizon_days, n_slope, n_target, haircut)
        except Exception:
            # Never crash a case: fall back to the last rate, starter spread.
            lr = last_rate(reads, idx)
            lr = lr if lr is not None else 0.01
            lg = logit(lr)
            spread = 0.5 * math.sqrt(horizon_days)
            qs = [round4(inv_logit(lg + z * spread)) for z in Z]
            for i in range(1, len(qs)):
                if qs[i] < qs[i - 1]:
                    qs[i] = qs[i - 1]
        rows.append({"id": trophy.get("id"), "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
