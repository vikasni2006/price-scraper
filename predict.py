"""Buy-now-or-wait prediction for one product.

It looks at four things and weighs them:
  1. Trend    - a straight-line fit over the last 30 days of prices, used to
                forecast the price 7 days ahead.
  2. Position - where today's price sits compared with every price we have
                recorded (lowest, average, highest).
  3. Discount - how far the price is below the MRP the store shows.
  4. Season   - big Indian sale seasons happen around the same time every year.

Each signal pushes towards "buy" or "wait". The totals decide the verdict, and
how much history there is decides how confident we are.
"""
from datetime import date, timedelta

TREND_DAYS = 30        # how far back the trend line looks
FORECAST_DAYS = 7      # how far ahead it forecasts
VERDICT_AT = 0.8       # score needed to say "buy" or "wait" instead of "fair"

# Approximate windows; the exact dates move a little each year.
SALE_SEASONS = [
    ((1, 10), (1, 26), "Republic Day sales"),
    ((7, 10), (7, 25), "July sales such as Prime Day"),
    ((8, 1), (8, 15), "Independence Day sales"),
    ((9, 20), (11, 10), "festive season sales (Big Billion Days, Great Indian Festival)"),
]


def rupee(n):
    return f"₹{n:,.0f}"


def sale_status(today):
    """('on', name, 0) during a usual sale window, ('soon', name, days) when one
    usually starts within 30 days, otherwise None."""
    upcoming = []
    for (sm, sd), (em, ed), name in SALE_SEASONS:
        for year in (today.year, today.year + 1):
            start, end = date(year, sm, sd), date(year, em, ed)
            if start <= today <= end:
                return ("on", name, 0)
            if today < start:
                upcoming.append(((start - today).days, name))
    days, name = min(upcoming)
    return ("soon", name, days) if days <= 30 else None


def fit_line(points):
    """Least-squares line through (x, y) points. Returns slope, intercept, r2."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in points)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return slope, intercept, max(r2, 0.0)


def predict(history, mrp=None, today=None):
    """history: list of (iso_day, price), oldest first, the last one is today's price.
    Returns a dict the web page can show directly."""
    today = today or date.today()
    history = [(date.fromisoformat(d) if isinstance(d, str) else d, float(p)) for d, p in history]
    history.sort()
    current = history[-1][1]
    prices = [p for _, p in history]
    n = len(history)
    first_day = history[0][0]
    span = (history[-1][0] - first_day).days
    low, high = min(prices), max(prices)
    avg = sum(prices) / n

    buy, wait = 0.0, 0.0
    reasons = []          # (direction, text) - direction is "buy", "wait" or "info"
    forecast = None
    trend_pct = None
    r2 = 0.0

    # 1. Trend: the direction of the fitted line, and whether today jumped off it
    direction = None      # "falling", "rising" or "steady"
    jump = None           # "spike" (today well above the line) or "dip" (well below)
    recent = [(d, p) for d, p in history if (history[-1][0] - d).days <= TREND_DAYS]
    if len(recent) >= 3 and (recent[-1][0] - recent[0][0]).days >= 3:
        origin = recent[0][0]
        pts = [((d - origin).days, p) for d, p in recent]
        slope, intercept, r2 = fit_line(pts)
        line_now = max(slope * pts[-1][0] + intercept, 1)
        forecast = slope * (pts[-1][0] + FORECAST_DAYS) + intercept
        forecast = min(max(forecast, low * 0.6), high * 1.4)  # keep it believable
        trend_pct = (forecast - current) / current * 100
        weekly_pct = slope * FORECAST_DAYS / line_now * 100   # how fast the line moves
        off_line = (current - line_now) / line_now * 100       # how far today is from it
        weight = min(abs(weekly_pct) / 3, 2.5) * (0.5 + 0.5 * r2)

        if weekly_pct <= -3:
            direction = "falling"
            wait += weight
            reasons.append(("wait", f"The price has been falling about {abs(weekly_pct):.0f}% a week. "
                                    f"At this rate it should be about {rupee(forecast)} in a week."))
        elif weekly_pct >= 3:
            direction = "rising"
            buy += weight
            reasons.append(("buy", f"The price has been rising about {weekly_pct:.0f}% a week. "
                                   f"At this rate it should be about {rupee(forecast)} in a week."))
        else:
            direction = "steady"
            reasons.append(("info", f"The price has been fairly steady. We expect about "
                                    f"{rupee(forecast)} next week."))

        if off_line >= 5:
            jump = "spike"
            wait += 0.8
            reasons.append(("wait", f"Today's price jumped {off_line:.0f}% above its recent level. "
                                    "Jumps like this often come back down."))
        elif off_line <= -5:
            jump = "dip"
            buy += 0.8
            reasons.append(("buy", f"Today's price dropped {abs(off_line):.0f}% below its recent level. "
                                   "Dips like this often don't last."))

    # 2. Position against everything we've recorded.
    # Being at the lowest is expected while a price falls, and being above average
    # is expected while it rises, so those cases don't count twice.
    if n >= 3:
        if current <= low * 1.01:
            if direction == "falling":
                reasons.append(("info", "It's the lowest price we've recorded, but it's still going down."))
            else:
                buy += 1.5
                reasons.append(("buy", f"This is the lowest price we've recorded since "
                                       f"{first_day.strftime('%d %b')}."))
        elif current >= avg * 1.08:
            if direction == "rising" and jump != "spike":
                reasons.append(("info", f"It's above its usual price of {rupee(avg)} because it has "
                                        "been going up."))
            else:
                wait += 1.0
                reasons.append(("wait", f"It's {(current / avg - 1) * 100:.0f}% above its usual price of "
                                        f"{rupee(avg)}. It has been as low as {rupee(low)}."))
        elif current <= avg * 0.95:
            buy += 0.7
            reasons.append(("buy", f"It's below its usual price of {rupee(avg)}."))
        else:
            reasons.append(("info", f"It's close to its usual price of {rupee(avg)}. "
                                    f"The lowest we've seen is {rupee(low)}."))

    # 3. Discount against MRP
    discount = None
    if mrp and mrp > current:
        discount = round((mrp - current) / mrp * 100)
        # MRPs are often set high on purpose, so a big discount is only a weak signal
        if discount >= 50:
            buy += 0.4
            reasons.append(("buy", f"It's {discount}% below the MRP of {rupee(mrp)}, a deep discount."))
        elif discount >= 25:
            buy += 0.2
            reasons.append(("buy", f"It's {discount}% below the MRP of {rupee(mrp)}."))
        elif discount < 10:
            wait += 0.4
            reasons.append(("wait", f"It's only {discount}% below the MRP, so there's room for a bigger discount."))

    # 4. Sale season
    season = sale_status(today)
    at_low = n >= 3 and current <= low * 1.01
    if season and season[0] == "on":
        buy += 0.3
        reasons.append(("buy", f"It's the time of year for {season[1]}, when prices are often at their lowest."))
    elif season and not at_low:
        wait += 0.8 if season[2] <= 14 else 0.5
        reasons.append(("wait", f"{season[1][0].upper() + season[1][1:]} usually start in about "
                                f"{season[2]} days, and prices often drop then."))

    # Verdict
    score = wait - buy
    if n == 1 and not reasons:
        verdict = "unknown"
    elif score >= VERDICT_AT:
        verdict = "wait"
    elif score <= -VERDICT_AT:
        verdict = "buy"
    else:
        verdict = "fair"

    if n < 3 or span < 3:
        confidence = "low"
    elif n < 10 or (trend_pct is not None and r2 < 0.5):
        confidence = "medium"
    else:
        confidence = "high"

    falling = direction == "falling" or jump == "spike"
    if verdict == "buy":
        label, headline, when = "Good time to buy", "Good time to buy", "Now"
    elif verdict == "wait":
        label = "Likely to drop" if falling else "Better to wait"
        headline = "Wait, the price is likely to drop" if falling else "Better to wait"
        if falling:
            when = "In about a week"
        elif season and season[0] == "soon":
            when = f"During {season[1]}, in about {season[2]} days"
        else:
            when = f"When it's back near {rupee(low)}"
    elif verdict == "fair":
        label, headline, when = "Fair price", "Fair price, no strong reason to wait", "Now, if you need it"
    else:
        label, headline, when = "Tracking started", "Not enough price history yet", "Check back in a few days"

    if n < 3:
        reasons.append(("info", f"We've recorded this price on {n} day{'s' if n != 1 else ''}. "
                                "Predictions get better after a few days of tracking."))

    return {
        "verdict": verdict,
        "label": label,
        "headline": headline,
        "when": when,
        "confidence": confidence,
        "forecast": round(forecast) if forecast is not None else None,
        "forecast_day": (history[-1][0] + timedelta(days=FORECAST_DAYS)).isoformat() if forecast is not None else None,
        "trend_pct": round(trend_pct, 1) if trend_pct is not None else None,
        "discount": discount,
        "reasons": [{"type": t, "text": s} for t, s in reasons],
        "stats": {
            "current": current, "low": low, "high": high, "avg": round(avg),
            "days": n, "since": first_day.isoformat(),
        },
    }
