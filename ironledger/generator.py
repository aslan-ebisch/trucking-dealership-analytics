"""Synthetic data engine for the fictional dealership group.

Design principles
-----------------
* **One unit master.** Every physical truck/trailer gets exactly one row in the
  ``inventory`` table. Sold units carry their ``sale_id``; unsold units are the
  current stock. Inventory and sales therefore cannot contradict each other.
* **Causal pricing.** Prices come from a model (base price x spec x market-cycle
  index x depreciation x mileage). Used-vehicle acquisition cost is set from the
  market value *on the acquisition date*, so when the market falls while a unit
  sits on the lot, the dealer loses money — exactly as in real life.
* **Exact accounting identities.** All money is stored as whole dollars (int64)
  and gross profit is computed by subtraction, so identities hold exactly.
* **Deterministic.** A single ``numpy.random.Generator`` seeded once drives every
  draw, in a fixed order. Same seed + same parameters => identical data.
* **Vectorized.** Python loops only run over small groups (stores, segments),
  never over individual transactions, so 100k+ sales generate in seconds.

All entities are fictional. See ``reference_data`` for the assumptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ironledger import __version__
from ironledger import reference_data as ref
from ironledger.config import AS_OF_DATE, DEFAULT_SEED, HISTORY_START, TABLES


@dataclass(frozen=True)
class GeneratorConfig:
    n_sales: int = 30_000
    n_customers: int = 5_000
    n_salespeople: int = 100
    seed: int = DEFAULT_SEED
    start: str = str(HISTORY_START.date())
    as_of: str = str(AS_OF_DATE.date())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sample_rows(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one column index per row of a (n x k) non-negative weight matrix."""
    cum = np.cumsum(probs, axis=1)
    u = rng.random(len(probs))[:, None] * cum[:, -1:]
    return (u < cum).argmax(axis=1)


def _round_to(x: np.ndarray, step: int) -> np.ndarray:
    return (np.round(np.asarray(x, dtype=float) / step) * step).astype(np.int64)


def _days(a, b) -> np.ndarray:
    """Whole days from a to b (b - a) for datetime64 arrays."""
    return ((np.asarray(b, dtype="datetime64[D]") - np.asarray(a, dtype="datetime64[D]"))
            .astype(np.int64))


def _allocate(total: int, weights: np.ndarray, minimum: int) -> np.ndarray:
    """Split an integer total proportionally to weights (largest remainder), with a floor."""
    weights = np.asarray(weights, dtype=float)
    base = np.full(len(weights), minimum)
    remaining = total - base.sum()
    raw = remaining * weights / weights.sum()
    alloc = np.floor(raw).astype(int)
    leftover = remaining - alloc.sum()
    order = np.argsort(-(raw - alloc), kind="stable")
    alloc[order[:leftover]] += 1
    return base + alloc


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------
def build_dealerships() -> pd.DataFrame:
    rows = []
    for d in ref.DEALERSHIPS:
        rows.append({
            "dealership_id": d["dealership_id"],
            "dealership_name": d["dealership_name"],
            "city": d["city"],
            "state": d["state"],
            "region": d["region"],
            "latitude": d["latitude"],
            "longitude": d["longitude"],
            "opened_date": pd.Timestamp(d["opened_date"]),
            "truck_franchises": "; ".join(d["franchises"]),
            "trailer_lines": "; ".join(d["trailer_lines"]),
            "lot_capacity": d["lot_capacity"],
            "volume_weight": d["volume_weight"],
        })
    return pd.DataFrame(rows)


def build_salespeople(rng, dealers: pd.DataFrame, n_total: int, start: pd.Timestamp,
                      as_of: pd.Timestamp) -> pd.DataFrame:
    """Sales roster with hiring, turnover and replacements.

    The network has a fixed number of sales "seats". Some seats turn over during the
    history (rep leaves, a replacement is hired), which brings the roster to n_total.
    Seats 0 and 1 at each store never turn over, guaranteeing coverage every day.
    """
    n_repl = int(round(n_total * 0.18))
    n_seats = n_total - n_repl
    seats = _allocate(n_seats, dealers["volume_weight"].to_numpy(), minimum=3)

    first, last = np.array(ref.FIRST_NAMES), np.array(ref.LAST_NAMES)
    combos = rng.choice(len(first) * len(last), size=n_total, replace=False)
    names = [f"{first[c // len(last)]} {last[c % len(last)]}" for c in combos]

    records = []
    for d_idx, row in dealers.iterrows():
        opened = row["opened_date"]
        for s in range(seats[d_idx]):
            if opened < start:
                if s >= 2 and rng.random() < 0.22:  # growth hire during the history
                    hire = start + pd.Timedelta(days=int(rng.integers(60, 1700)))
                else:
                    hire = max(opened, pd.Timestamp("2006-01-01")) + pd.Timedelta(
                        days=int(rng.integers(0, max(30, (start - opened).days - 30))))
            else:  # store opened during the history: staff hired just before opening
                hire = opened - pd.Timedelta(days=int(rng.integers(7, 75)))
            records.append({"dealership_id": row["dealership_id"], "seat": s, "hire_date": hire,
                            "termination_date": pd.NaT})
    roster = pd.DataFrame(records)

    # Turnover: pick seats (never seat 0/1) whose rep leaves and is replaced.
    eligible = roster.index[(roster["seat"] >= 2) & (roster["hire_date"] < as_of - pd.Timedelta(days=500))]
    chosen = rng.choice(eligible, size=min(n_repl, len(eligible)), replace=False)
    replacements = []
    for i in np.sort(chosen):
        hire = roster.at[i, "hire_date"]
        lo = max(hire + pd.Timedelta(days=200), start + pd.Timedelta(days=60))
        hi = as_of - pd.Timedelta(days=150)
        term = lo + pd.Timedelta(days=int(rng.integers(0, max(1, (hi - lo).days))))
        roster.at[i, "termination_date"] = term
        replacements.append({"dealership_id": roster.at[i, "dealership_id"], "seat": roster.at[i, "seat"],
                             "hire_date": term + pd.Timedelta(days=int(rng.integers(10, 60))),
                             "termination_date": pd.NaT})
    roster = pd.concat([roster, pd.DataFrame(replacements)], ignore_index=True)
    roster = roster.sort_values(["dealership_id", "hire_date", "seat"], kind="stable").reset_index(drop=True)

    # Roles: the longest-tenured active rep at larger stores runs fleet accounts.
    roster["role"] = np.where(roster["hire_date"] < pd.Timestamp("2018-01-01"),
                              "Senior Sales Consultant", "Sales Consultant")
    for d_id, grp in roster.groupby("dealership_id"):
        n_fam = 2 if dealers.set_index("dealership_id").at[d_id, "volume_weight"] >= 0.85 else 1
        fam_idx = grp[grp["seat"] < 2].index[:n_fam]
        roster.loc[fam_idx, "role"] = "Fleet Account Manager"

    roster["salesperson_id"] = [f"SP{i + 1:03d}" for i in range(len(roster))]
    roster["salesperson_name"] = names[: len(roster)]
    roster["employment_status"] = np.where(roster["termination_date"].isna(), "Active", "Former")
    roster["skill"] = np.round(rng.lognormal(0.0, 0.35, len(roster)), 3)  # hidden productivity driver
    return roster[["salesperson_id", "salesperson_name", "dealership_id", "role", "hire_date",
                   "termination_date", "employment_status", "skill"]]


def build_customers(rng, dealers: pd.DataFrame, n: int) -> pd.DataFrame:
    seg_idx = rng.choice(len(ref.CUSTOMER_SEGMENTS), size=n, p=ref.SEGMENT_CUSTOMER_SHARE)
    segments = np.array(ref.CUSTOMER_SEGMENTS)[seg_idx]

    # Home dealership: store size x local demand for that segment.
    d_ids = dealers["dealership_id"].to_numpy()
    tilt = np.array([[d["segment_tilt"].get(s, 1.0) for s in ref.CUSTOMER_SEGMENTS] for d in ref.DEALERSHIPS])
    w = dealers["volume_weight"].to_numpy()[:, None] * tilt  # (stores x segments)
    probs = w[:, seg_idx].T
    home = d_ids[_sample_rows(rng, probs)]

    home_state = np.empty(n, dtype=object)
    for i in range(n):
        pool = ref.NATIONAL_FLEET_HQ_STATES if segments[i] == "National Fleet" else ref.STORE_CUSTOMER_STATES[home[i]]
        home_state[i] = pool[rng.integers(len(pool))]

    names, used = [], set()
    for seg in segments:
        for attempt in range(50):
            core = ref.COMPANY_CORES[seg][rng.integers(len(ref.COMPANY_CORES[seg]))]
            if seg == "Owner-Operator":
                base = f"{ref.LAST_NAMES[rng.integers(len(ref.LAST_NAMES))]} {core}"
            else:
                base = f"{ref.COMPANY_PREFIXES[rng.integers(len(ref.COMPANY_PREFIXES))]} {core}"
            if seg == "Government & Municipal":
                name = base if attempt < 25 else f"{base} No. {attempt}"
            else:
                name = f"{base} {ref.COMPANY_SUFFIXES[rng.integers(len(ref.COMPANY_SUFFIXES))]}"
                if attempt >= 25:
                    name = f"{base} {attempt} {ref.COMPANY_SUFFIXES[0]}"
            if name not in used:
                break
        used.add(name)
        names.append(name)

    lo = np.array([ref.SEGMENT_FLEET_SIZE[s][0] for s in segments], dtype=float)
    hi = np.array([ref.SEGMENT_FLEET_SIZE[s][1] for s in segments], dtype=float)
    fleet = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi + 0.999)))).astype(int)

    tier_p = {"National Fleet": [.80, .18, .02, 0], "Government & Municipal": [.90, .10, 0, 0],
              "Regional Carrier": [.50, .35, .12, .03]}
    tiers = np.array([rng.choice(["A", "B", "C", "D"], p=tier_p.get(s, [.30, .38, .22, .10])) for s in segments])

    propensity = np.array([ref.SEGMENT_PROPENSITY[s] for s in segments]) * rng.lognormal(0, 0.9, n)

    return pd.DataFrame({
        "customer_id": [f"C{i + 1:05d}" for i in range(n)],
        "customer_name": names,
        "customer_segment": segments,
        "home_state": home_state,
        "primary_dealership_id": home,
        "fleet_size": fleet,
        "credit_tier": tiers,
        "_propensity": propensity,
    })


# ---------------------------------------------------------------------------
# Vehicle valuation model
# ---------------------------------------------------------------------------
def _depreciation(age_years: np.ndarray, is_trailer: np.ndarray) -> np.ndarray:
    """Fraction of new value retained at a given age (before market & mileage effects)."""
    a = np.maximum(age_years, 0.0)
    truck = np.where(a < 1, 1 - 0.20 * a, 0.80 * 0.865 ** (a - 1))
    trailer = np.where(a < 1, 1 - 0.15 * a, 0.85 * 0.91 ** (a - 1))
    return np.maximum(np.where(is_trailer, trailer, truck), 0.07)


def _used_retail_value(base, spec, price_factor, model_year, mileage_adj, is_trailer, when) -> np.ndarray:
    """Retail market value of a used unit on date ``when``."""
    when = pd.to_datetime(when)
    age = (when - pd.to_datetime(model_year.astype(str) + "-07-01")).days.to_numpy() / 365.25
    return (base * spec * price_factor
            * ref.market_index(when, "new_price")
            * _depreciation(age, is_trailer)
            * mileage_adj
            * ref.market_index(when, "used_value"))


def _attach_units(rng, n: int, store_idx: np.ndarray, is_trailer: np.ndarray, is_new: np.ndarray,
                  catalog: pd.DataFrame, cat_factor: np.ndarray) -> np.ndarray:
    """Choose a catalog row for each unit given store, vehicle type and condition."""
    out = np.empty(n, dtype=np.int64)
    cat_codes = catalog["vehicle_category"].map({c: i for i, c in enumerate(ref.VEHICLE_CATEGORIES)}).to_numpy()
    cat_is_trailer = (catalog["vehicle_type"] == "Trailer").to_numpy()
    for s, d in enumerate(ref.DEALERSHIPS):
        for trailer in (False, True):
            for new in (True, False):
                mask = (store_idx == s) & (is_trailer == trailer) & (is_new == new)
                k = int(mask.sum())
                if k == 0:
                    continue
                brands = d["trailer_lines"] if trailer else d["franchises"]
                in_line = catalog["manufacturer"].isin(brands).to_numpy()
                eligible = cat_is_trailer == trailer
                if new:
                    eligible &= in_line
                w = catalog["sample_weight"].to_numpy() * cat_factor[s, cat_codes]
                if not new:  # used stock skews toward the store's own brands (trade-ins)
                    w = w * np.where(in_line, 2.5, 1.0)
                w = np.where(eligible, w, 0.0)
                out[mask] = rng.choice(len(catalog), size=k, p=w / w.sum())
    return out


def _store_category_factor() -> np.ndarray:
    """Category demand multiplier per store, implied by its customer-segment tilt."""
    mix = np.array([ref.CATEGORY_SEGMENT_MIX[c] for c in ref.VEHICLE_CATEGORIES])  # (cat x seg)
    tilt = np.array([[d["segment_tilt"].get(s, 1.0) for s in ref.CUSTOMER_SEGMENTS] for d in ref.DEALERSHIPS])
    return (tilt @ mix.T) / mix.sum(axis=1)[None, :]  # (store x cat)


def _build_units(rng, *, n, dates, store_idx, is_trailer, is_new, catalog, cat_factor, dealers,
                 value_date, acq_dates, market_date_for_new):
    """Shared unit attributes: model, year, mileage, costs. ``value_date`` is the
    date at which a selling/list price is struck (sale date, or as-of for stock)."""
    price_factor = np.array([d["price_factor"] for d in ref.DEALERSHIPS])[store_idx]
    row = _attach_units(rng, n, store_idx, is_trailer, is_new, catalog, cat_factor)
    c = catalog.iloc[row].reset_index(drop=True)
    base = c["base_price"].to_numpy(dtype=float)
    spec = rng.lognormal(0.0, 0.06, n)  # option content / spec variance

    acq_year = pd.DatetimeIndex(acq_dates).year.to_numpy()
    acq_month = pd.DatetimeIndex(acq_dates).month.to_numpy()
    ref_year = pd.DatetimeIndex(market_date_for_new).year.to_numpy()
    ref_month = pd.DatetimeIndex(market_date_for_new).month.to_numpy()

    # Model year ---------------------------------------------------------------
    next_my = (ref_month >= 8) & (rng.random(n) < 0.55)
    new_my = np.where(is_new, np.minimum(ref_year, acq_year + 1) + next_my, 0)
    age_draw = np.where(is_trailer, 1 + rng.gamma(3.0, 1.8, n), 1 + rng.gamma(3.0, 1.35, n))
    age_draw = np.clip(np.round(age_draw), 1, np.where(is_trailer, 18, 14)).astype(int)
    used_my = acq_year - age_draw + (acq_month >= 9)
    model_year = np.where(is_new, new_my, used_my).astype(int)

    # Mileage (at the value date) ---------------------------------------------
    vd = pd.DatetimeIndex(value_date)
    age_frac = np.maximum((vd - pd.to_datetime(model_year.astype(str) + "-07-01")).days.to_numpy() / 365.25, 0.4)
    annual = c["vehicle_category"].map(ref.ANNUAL_MILES).fillna(0).to_numpy()
    expected = annual * age_frac
    used_miles = expected * rng.lognormal(0.0, 0.28, n)
    new_miles = rng.integers(15, 450, n)
    mileage = np.where(is_new, new_miles, np.round(used_miles, -2)).astype(float)
    mileage[is_trailer] = np.nan  # trailers have no odometer reading
    mileage_adj = np.where(is_new | is_trailer, 1.0,
                           np.clip(1 - 0.22 * (np.nan_to_num(mileage) / np.maximum(expected, 1) - 1), 0.75, 1.12))

    # Selling / list price -------------------------------------------------------
    my_premium = np.where(model_year > pd.DatetimeIndex(value_date).year, 1.015, 1.0)
    new_price = base * spec * price_factor * ref.market_index(market_date_for_new, "new_price") * my_premium
    used_value_at_sale = _used_retail_value(base, spec, price_factor, model_year, mileage_adj, is_trailer, value_date)
    used_value_at_acq = _used_retail_value(base, spec, price_factor, model_year, mileage_adj, is_trailer, acq_dates)

    # Costs ---------------------------------------------------------------------
    recon_new = np.where(is_trailer, rng.uniform(150, 500, n), rng.uniform(350, 1400, n))
    age_now = np.maximum(age_frac, 0)
    recon_used = np.where(
        is_trailer,
        np.clip(rng.lognormal(np.log(1100), 0.6, n), 100, 8_000),
        np.clip(rng.lognormal(np.log(3800), 0.6, n) * (1 + 0.05 * age_now), 300, 30_000),
    )
    reconditioning = np.round(np.where(is_new, recon_new, recon_used)).astype(np.int64)

    src_new = np.where(rng.random(n) < 0.85, "Factory Order", "Dealer Trade")
    src_used = rng.choice(ref.ACQUISITION_SOURCES_USED, size=n, p=ref.ACQUISITION_SOURCES_USED_P)
    source = np.where(is_new, src_new, src_used)
    transport = np.select(
        [is_new & ~is_trailer, is_new & is_trailer,
         source == "Trade-In", source == "Fleet Buyback", source == "Auction"],
        [rng.uniform(1200, 3600, n), rng.uniform(500, 1500, n),
         np.where(rng.random(n) < 0.12, rng.uniform(150, 600, n), 0.0),
         rng.uniform(300, 1800, n), rng.uniform(500, 2600, n)],
        default=rng.uniform(400, 2200, n),
    )
    transport = np.round(transport).astype(np.int64)

    spread = np.clip(rng.normal(np.where(is_trailer, 0.20, 0.19), 0.045, n), 0.05, 0.34)
    used_acq_cost = used_value_at_acq * (1 - spread)

    return dict(row=row, c=c, base=base, spec=spec, model_year=model_year, mileage=mileage,
                new_price=new_price, used_value_at_sale=used_value_at_sale, used_acq_cost=used_acq_cost,
                reconditioning=reconditioning, transport=transport, source=source)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_all(cfg: GeneratorConfig = GeneratorConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    start, as_of = pd.Timestamp(cfg.start), pd.Timestamp(cfg.as_of)
    catalog = ref.product_catalog()
    dealers = build_dealerships()
    n_stores = len(dealers)
    cat_factor = _store_category_factor()
    store_vw = dealers["volume_weight"].to_numpy()
    store_open = dealers["opened_date"].to_numpy(dtype="datetime64[D]")

    salespeople = build_salespeople(rng, dealers, cfg.n_salespeople, start, as_of)
    customers = build_customers(rng, dealers, cfg.n_customers)

    # 1. Sale dates & stores ----------------------------------------------------
    days = pd.date_range(start, as_of, freq="D")
    n_days = len(days)
    day_w = (ref.market_index(days, "volume")
             * ref.MONTH_SEASONALITY[days.month.to_numpy() - 1]
             * ref.WEEKDAY_FACTOR[days.dayofweek.to_numpy()])
    days_since_open = _days(store_open[:, None], days.to_numpy()[None, :])  # (store x day)
    ramp = np.clip(0.45 + 0.55 * days_since_open / 270.0, 0, 1)  # new stores ramp up over ~9 months
    cell_w = store_vw[:, None] * day_w[None, :] * np.where(days_since_open >= 0, ramp, 0.0)
    cell = rng.choice(cell_w.size, size=cfg.n_sales, p=(cell_w / cell_w.sum()).ravel())
    store_idx, day_idx = np.divmod(cell, n_days)
    order = np.lexsort((store_idx, day_idx))
    store_idx, day_idx = store_idx[order], day_idx[order]
    sale_date = days.to_numpy()[day_idx]
    n = cfg.n_sales

    # 2. Vehicle type & condition (market-dependent) -----------------------------
    vol = ref.market_index(sale_date, "volume")
    tight = ref.market_index(sale_date, "new_supply_tight")
    p_trailer = np.clip(0.23 * (1 + 1.2 * (vol - 1)), 0.12, 0.35)
    is_trailer = rng.random(n) < p_trailer
    p_new = np.where(is_trailer, 0.66 - 0.06 * tight, 0.54 - 0.08 * tight)
    is_new = rng.random(n) < p_new

    # 3. Days in inventory -> acquisition date ----------------------------------
    used_now = ref.market_index(sale_date, "used_value")
    used_90 = ref.market_index(sale_date - np.timedelta64(90, "D"), "used_value")
    used_trend = used_now / used_90 - 1
    median_days = np.select(
        [is_new & ~is_trailer, is_new & is_trailer, ~is_new & ~is_trailer],
        [48 * np.exp(-0.55 * tight), 42 * np.exp(-0.30 * tight), 58 * np.exp(-2.5 * used_trend)],
        default=50 * np.exp(-1.5 * used_trend),
    )
    dii = np.clip(np.round(median_days * rng.lognormal(0, 0.75, n)), 1, 720).astype(np.int64)
    earliest = store_open[store_idx] - np.timedelta64(60, "D")  # pre-opening stocking allowed
    dii = np.minimum(dii, _days(earliest, sale_date))
    acq_date = (sale_date.astype("datetime64[D]") - dii.astype("timedelta64[D]")).astype("datetime64[ns]")

    # 4. Units, prices and costs -------------------------------------------------
    u = _build_units(rng, n=n, dates=sale_date, store_idx=store_idx, is_trailer=is_trailer, is_new=is_new,
                     catalog=catalog, cat_factor=cat_factor, dealers=dealers,
                     value_date=sale_date, acq_dates=acq_date, market_date_for_new=sale_date)
    store_margin = np.array([d["margin_adj"] for d in ref.DEALERSHIPS])[store_idx]

    used_price = u["used_value_at_sale"] * rng.normal(1.0, 0.035, n)
    sale_price = np.where(is_new, _round_to(u["new_price"], 100), np.maximum(_round_to(used_price, 100), 4_000))

    aging_penalty = 0.00012 * np.maximum(dii - 120, 0)
    new_margin = np.clip(
        np.where(is_trailer, 0.072 + 0.010 * tight, 0.064 + 0.028 * tight)
        + store_margin - aging_penalty + rng.normal(0, 0.022, n),
        -0.08, 0.20,
    )
    new_acq = sale_price * (1 - new_margin) - u["reconditioning"] - u["transport"]
    acquisition_cost = np.round(np.where(is_new, new_acq, np.maximum(u["used_acq_cost"], 1_000))).astype(np.int64)
    reconditioning, transport = u["reconditioning"], u["transport"]
    sale_price = sale_price.astype(np.int64)
    gross = sale_price - acquisition_cost - reconditioning - transport

    # 5. Customers ----------------------------------------------------------------
    cat_names = u["c"]["vehicle_category"].to_numpy()
    cat_code = pd.Categorical(cat_names, categories=ref.VEHICLE_CATEGORIES).codes
    mix = np.array([ref.CATEGORY_SEGMENT_MIX[c] for c in ref.VEHICLE_CATEGORIES])
    tilt = np.array([[d["segment_tilt"].get(s, 1.0) for s in ref.CUSTOMER_SEGMENTS] for d in ref.DEALERSHIPS])
    seg_idx = _sample_rows(rng, mix[cat_code] * tilt[store_idx])
    segment = np.array(ref.CUSTOMER_SEGMENTS)[seg_idx]

    d_ids = dealers["dealership_id"].to_numpy()
    cust_ids = customers["customer_id"].to_numpy()
    customer_id = np.empty(n, dtype=object)
    for s in range(n_stores):
        for g, seg in enumerate(ref.CUSTOMER_SEGMENTS):
            mask = (store_idx == s) & (seg_idx == g)
            k = int(mask.sum())
            if k == 0:
                continue
            pool = customers["customer_segment"].to_numpy() == seg
            home = customers["primary_dealership_id"].to_numpy() == d_ids[s]
            cross = 0.35 if seg == "National Fleet" else 0.02
            w = customers["_propensity"].to_numpy() * np.where(home, 1.0, cross) * pool
            if w.sum() == 0:  # tiny datasets may lack a segment entirely: fall back to anyone
                w = customers["_propensity"].to_numpy().copy()
            customer_id[mask] = cust_ids[rng.choice(len(customers), size=k, p=w / w.sum())]

    # 6. Sale channel -------------------------------------------------------------
    ch_mix = np.array([ref.SEGMENT_CHANNEL_MIX[s] for s in ref.CUSTOMER_SEGMENTS])
    channel = np.array(ref.SALE_CHANNELS)[_sample_rows(rng, ch_mix[seg_idx])]
    progress = _days(start.to_datetime64(), sale_date) / max(1, (as_of - start).days)
    shift_online = (channel == "Showroom / Walk-In") & (rng.random(n) < 0.04 + 0.16 * progress)
    channel = np.where(shift_online, "Online Lead", channel)

    # 7. Salesperson (only reps employed at that store on the sale date) ---------
    sp = salespeople
    hire_i = _days(start.to_datetime64(), sp["hire_date"].to_numpy())
    term_i = np.where(sp["termination_date"].isna(), n_days + 1,
                      _days(start.to_datetime64(), sp["termination_date"].fillna(as_of).to_numpy()))
    is_fam = (sp["role"] == "Fleet Account Manager").to_numpy()
    fleet_deal = (channel == "Fleet Account") | np.isin(segment, ["National Fleet", "Regional Carrier"])
    salesperson_id = np.empty(n, dtype=object)
    for s in range(n_stores):
        reps = np.flatnonzero(sp["dealership_id"].to_numpy() == d_ids[s])
        sel = np.flatnonzero(store_idx == s)
        if len(sel) == 0:
            continue
        di = day_idx[sel][:, None]
        active = (di >= hire_i[reps][None, :]) & (di <= term_i[reps][None, :])
        w = active * sp["skill"].to_numpy()[reps][None, :]
        boost = np.where(fleet_deal[sel][:, None], np.where(is_fam[reps], 4.0, 1.0),
                         np.where(is_fam[reps], 0.6, 1.0))
        salesperson_id[sel] = sp["salesperson_id"].to_numpy()[reps][_sample_rows(rng, w * boost)]

    # 8. F&I: financing and protection products ----------------------------------
    fin_rate = np.array([ref.SEGMENT_FINANCE_RATE[s] for s in segment]) * np.where(is_new, 1.0, 1.12)
    financing = rng.random(n) < np.clip(fin_rate, 0, 0.95)
    amount_financed = np.where(financing, np.round(sale_price * rng.uniform(0.75, 0.92, n)), 0).astype(np.int64)
    finance_income = np.where(financing, np.minimum(np.round(amount_financed * rng.uniform(0.009, 0.022, n)), 6_500),
                              0).astype(np.int64)
    prot_rate = (np.array([ref.SEGMENT_PROTECTION_RATE[s] for s in segment])
                 * np.where(is_new, 1.0, 1.15) * np.where(is_trailer, 0.55, 1.0))
    has_prot = rng.random(n) < prot_rate
    truck_prod = rng.choice(ref.TRUCK_PROTECTION_PRODUCTS, size=n, p=[.35, .25, .25, .15])
    product = np.where(has_prot, np.where(is_trailer, ref.TRAILER_PROTECTION_PRODUCTS[0], truck_prod), None)
    warranty_income = np.where(
        has_prot,
        np.where(is_trailer, rng.lognormal(np.log(650), 0.3, n),
                 rng.lognormal(np.log(np.where(is_new, 2400, 2900)), 0.35, n)),
        0,
    )
    warranty_income = np.round(warranty_income).astype(np.int64)

    commission = np.round(np.maximum(np.where(is_trailer, 150, 300), 0.25 * np.maximum(gross, 0))).astype(np.int64)

    # 9. Transaction status ---------------------------------------------------------
    age_at_asof = _days(sale_date, as_of.to_datetime64())
    pending_window = np.where(segment == "Government & Municipal", 30, 10)
    pending = (age_at_asof < pending_window) & (rng.random(n) < 0.5)
    status = np.where(pending, "Pending Funding", "Funded")

    sales = pd.DataFrame({
        "sale_date": pd.to_datetime(sale_date),
        "dealership_id": d_ids[store_idx],
        "salesperson_id": salesperson_id,
        "customer_id": customer_id,
        "customer_segment": segment,
        "manufacturer": u["c"]["manufacturer"].to_numpy(),
        "model": u["c"]["model"].to_numpy(),
        "model_year": u["model_year"],
        "vehicle_category": cat_names,
        "vehicle_type": np.where(is_trailer, "Trailer", "Truck"),
        "condition": np.where(is_new, "New", "Used"),
        "mileage": u["mileage"],
        "acquisition_source": u["source"],
        "inventory_acquisition_date": pd.to_datetime(acq_date),
        "days_in_inventory": dii,
        "acquisition_cost": acquisition_cost,
        "reconditioning_cost": reconditioning,
        "transportation_cost": transport,
        "sale_price": sale_price,
        "vehicle_gross_profit": gross,
        "financing_arranged": financing,
        "amount_financed": amount_financed,
        "finance_income": finance_income,
        "protection_product": product,
        "warranty_income": warranty_income,
        "sales_commission": commission,
        "sale_channel": channel,
        "transaction_status": status,
    })
    sales["_list_price"] = np.nan

    # 10. Current stock (unsold units as of the as-of date) -------------------------
    stock = _generate_stock(rng, sales, store_idx, is_trailer, is_new, catalog, cat_factor, dealers, as_of)

    # 11. Unit master: assign unit IDs chronologically by acquisition ---------------
    sold_units = sales[["dealership_id", "manufacturer", "model", "model_year", "vehicle_category", "vehicle_type",
                        "condition", "mileage", "acquisition_source", "inventory_acquisition_date",
                        "acquisition_cost", "reconditioning_cost", "transportation_cost", "_list_price"]].copy()
    sold_units["_sale_row"] = np.arange(n)
    stock["_sale_row"] = -1
    units = pd.concat([sold_units, stock], ignore_index=True)
    units = units.sort_values(["inventory_acquisition_date", "dealership_id", "_sale_row"], kind="stable")
    units["unit_id"] = [f"U{i + 1:07d}" for i in range(len(units))]

    sales["sale_id"] = [f"S{i + 1:07d}" for i in range(n)]
    sold_map = units.loc[units["_sale_row"] >= 0, ["_sale_row", "unit_id"]].set_index("_sale_row")["unit_id"]
    sales["unit_id"] = sold_map.reindex(np.arange(n)).to_numpy()

    # Denormalized reporting attributes & derived metrics
    dl = dealers.set_index("dealership_id")
    sales["dealership_name"] = sales["dealership_id"].map(dl["dealership_name"])
    sales["city"] = sales["dealership_id"].map(dl["city"])
    sales["state"] = sales["dealership_id"].map(dl["state"])
    sales["vehicle_gross_margin"] = np.round(sales["vehicle_gross_profit"] / sales["sale_price"], 6)
    sales["total_gross_contribution"] = (sales["vehicle_gross_profit"] + sales["finance_income"]
                                         + sales["warranty_income"])

    inventory = units.rename(columns={"inventory_acquisition_date": "acquisition_date", "_list_price": "list_price"})
    inventory["total_cost_basis"] = (inventory["acquisition_cost"] + inventory["reconditioning_cost"]
                                     + inventory["transportation_cost"])
    sale_lookup = sales.set_index("unit_id")[["sale_id", "sale_date"]]
    inventory["sale_id"] = inventory["unit_id"].map(sale_lookup["sale_id"])
    inventory["sale_date"] = inventory["unit_id"].map(sale_lookup["sale_date"])
    inventory["status"] = np.where(inventory["sale_id"].isna(), "In Stock", "Sold")
    end = inventory["sale_date"].fillna(as_of)
    inventory["days_in_inventory"] = (end - inventory["acquisition_date"]).dt.days.astype(np.int64)
    inventory["list_price"] = inventory["list_price"].round(-2)

    inventory = inventory[[
        "unit_id", "dealership_id", "manufacturer", "model", "model_year", "vehicle_category", "vehicle_type",
        "condition", "mileage", "acquisition_source", "acquisition_date", "acquisition_cost", "reconditioning_cost",
        "transportation_cost", "total_cost_basis", "list_price", "status", "sale_id", "sale_date",
        "days_in_inventory",
    ]].reset_index(drop=True)

    sales = sales[[
        "sale_id", "sale_date", "dealership_id", "dealership_name", "city", "state", "salesperson_id",
        "customer_id", "customer_segment", "unit_id", "manufacturer", "model", "model_year", "vehicle_category",
        "vehicle_type", "condition", "mileage", "acquisition_source", "inventory_acquisition_date",
        "days_in_inventory", "acquisition_cost", "reconditioning_cost", "transportation_cost", "sale_price",
        "vehicle_gross_profit", "vehicle_gross_margin", "financing_arranged", "amount_financed", "finance_income",
        "protection_product", "warranty_income", "total_gross_contribution", "sales_commission", "sale_channel",
        "transaction_status",
    ]]

    customers = _finalize_customers(rng, customers, sales, as_of)
    opex = _generate_operating_expenses(dealers, sales, inventory, start, as_of)
    salespeople = salespeople.drop(columns=["skill"])

    return {
        "dealerships": dealers,
        "salespeople": salespeople,
        "customers": customers,
        "inventory": inventory,
        "sales": sales,
        "operating_expenses": opex,
    }


def _generate_stock(rng, sales, store_idx, is_trailer, is_new, catalog, cat_factor, dealers, as_of) -> pd.DataFrame:
    """Units on the lot at the as-of date, sized from each store's recent sales pace."""
    recent = (sales["sale_date"] > as_of - pd.Timedelta(days=90)).to_numpy()
    # Calibrated to the months' supply implied by the sold history (on-hand units at past
    # month-ends ÷ monthly pace), so the as-of snapshot continues the historical trend line.
    months_supply = {(False, True): 2.0, (False, False): 2.5, (True, True): 1.8, (True, False): 2.3}
    s_idx, s_trl, s_new = [], [], []
    for s in range(len(dealers)):
        for trailer in (False, True):
            for new in (True, False):
                pace = ((store_idx == s) & (is_trailer == trailer) & (is_new == new) & recent).sum() / 3.0
                k = int(round(pace * months_supply[(trailer, new)] * rng.uniform(0.85, 1.15)))
                s_idx += [s] * k
                s_trl += [trailer] * k
                s_new += [new] * k
    store = np.array(s_idx, dtype=int)
    trl = np.array(s_trl, dtype=bool)
    new = np.array(s_new, dtype=bool)
    k = len(store)

    med = np.select([new & ~trl, new & trl, ~new & ~trl], [41, 29, 42], default=35)
    age = np.clip(np.round(med * rng.lognormal(0, 0.85, k)), 0, 600).astype(np.int64)
    open_d = dealers["opened_date"].to_numpy(dtype="datetime64[D]")[store] - np.timedelta64(60, "D")
    age = np.minimum(age, _days(open_d, as_of.to_datetime64()))
    acq = (np.datetime64(as_of.date()) - age.astype("timedelta64[D]")).astype("datetime64[ns]")
    as_of_arr = np.full(k, as_of.to_datetime64())

    u = _build_units(rng, n=k, dates=as_of_arr, store_idx=store, is_trailer=trl, is_new=new, catalog=catalog,
                     cat_factor=cat_factor, dealers=dealers, value_date=as_of_arr, acq_dates=acq,
                     market_date_for_new=acq)
    tight_acq = ref.market_index(acq, "new_supply_tight")
    margin = np.where(trl, 0.072 + 0.010 * tight_acq, 0.064 + 0.028 * tight_acq)
    list_new = u["new_price"] * ref.market_index(as_of_arr, "new_price") / ref.market_index(acq, "new_price")
    new_cost = u["new_price"] * (1 - margin) - u["reconditioning"] - u["transport"]
    acq_cost = np.round(np.where(new, new_cost, np.maximum(u["used_acq_cost"], 1_000))).astype(np.int64)
    list_price = np.where(new, list_new, u["used_value_at_sale"] * 1.04)

    return pd.DataFrame({
        "dealership_id": dealers["dealership_id"].to_numpy()[store],
        "manufacturer": u["c"]["manufacturer"].to_numpy(),
        "model": u["c"]["model"].to_numpy(),
        "model_year": u["model_year"],
        "vehicle_category": u["c"]["vehicle_category"].to_numpy(),
        "vehicle_type": np.where(trl, "Trailer", "Truck"),
        "condition": np.where(new, "New", "Used"),
        "mileage": u["mileage"],
        "acquisition_source": u["source"],
        "inventory_acquisition_date": pd.to_datetime(acq),
        "acquisition_cost": acq_cost,
        "reconditioning_cost": u["reconditioning"],
        "transportation_cost": u["transport"],
        "_list_price": list_price,
    })


def _finalize_customers(rng, customers: pd.DataFrame, sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    first = sales.groupby("customer_id")["sale_date"].min()
    last = sales.groupby("customer_id")["sale_date"].max()
    c = customers.drop(columns=["_propensity"]).copy()
    c["first_purchase"] = c["customer_id"].map(first)
    c["last_purchase"] = c["customer_id"].map(last)
    lead = pd.to_timedelta(rng.integers(0, 900, len(c)), unit="D")
    prospect_since = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, (as_of - pd.Timestamp("2016-01-01")).days, len(c)), unit="D")
    c["customer_since"] = (c["first_purchase"] - lead).fillna(pd.Series(prospect_since, index=c.index))
    c["account_status"] = np.select(
        [c["first_purchase"].isna(), c["last_purchase"] >= as_of - pd.Timedelta(days=730)],
        ["Prospect", "Active"], default="Lapsed")
    return c[["customer_id", "customer_name", "customer_segment", "home_state", "primary_dealership_id",
              "fleet_size", "credit_tier", "customer_since", "account_status"]]


def _generate_operating_expenses(dealers, sales, inventory, start, as_of) -> pd.DataFrame:
    """Monthly sales-department operating expenses per store.

    Scope: the *vehicle sales department* only (service, parts and body shop are not
    modeled). Payroll includes the commissions actually earned on each month's deals,
    and floorplan interest is charged on the inventory actually on hand mid-month.
    """
    months = pd.date_range(start, as_of, freq="MS")
    mid = months + pd.Timedelta(days=14)
    rate = ref.market_index(mid, "floorplan_rate")
    sm = sales.assign(month=sales["sale_date"].dt.to_period("M").dt.to_timestamp())
    agg = sm.groupby(["dealership_id", "month"]).agg(revenue=("sale_price", "sum"),
                                                     commissions=("sales_commission", "sum"))

    acq = inventory["acquisition_date"].to_numpy()
    sold = inventory["sale_date"].fillna(pd.Timestamp("2100-01-01")).to_numpy()
    financed_cost = (inventory["acquisition_cost"] + inventory["transportation_cost"]).to_numpy(dtype=float)
    store_of_unit = inventory["dealership_id"].to_numpy()

    rows = []
    month_of_year = months.month.to_numpy()
    util_season = np.array([1.25, 1.15, 0.95, 0.85, 0.90, 1.10, 1.30, 1.30, 1.05, 0.85, 0.95, 1.20])
    for _, d in dealers.iterrows():
        units_here = store_of_unit == d["dealership_id"]
        on_hand = (acq[units_here][:, None] <= mid.to_numpy()[None, :]) & (sold[units_here][:, None] > mid.to_numpy()[None, :])
        balance = financed_cost[units_here] @ on_hand  # floorplan balance per month
        size = d["volume_weight"] / 0.5
        for m_i, month in enumerate(months):
            if month < d["opened_date"].to_period("M").to_timestamp():
                continue
            yrs = (month - start).days / 365.25
            infl = 1.035 ** yrs
            key = (d["dealership_id"], month)
            revenue = agg["revenue"].get(key, 0)
            commissions = agg["commissions"].get(key, 0)
            payroll = commissions + 48_000 * size ** 0.6 * infl
            facility = 32_000 * size ** 0.6 * 1.03 ** yrs
            marketing = 4_000 * infl + 0.0035 * revenue
            utilities = 4_500 * size ** 0.5 * util_season[month_of_year[m_i] - 1] * infl
            floorplan = balance[m_i] * rate[m_i] / 12
            other = 7_000 * size ** 0.5 * infl + 0.0015 * revenue
            vals = np.round([payroll, facility, marketing, utilities, floorplan, other]).astype(np.int64)
            rows.append({"dealership_id": d["dealership_id"], "month": month,
                         "payroll_expense": vals[0], "facility_expense": vals[1], "marketing_expense": vals[2],
                         "utilities_expense": vals[3], "floorplan_interest_expense": vals[4],
                         "other_operating_expense": vals[5], "total_operating_expense": int(vals.sum()),
                         "avg_floorplan_balance": int(round(balance[m_i])),
                         "floorplan_rate": round(float(rate[m_i]), 5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def existing_outputs(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    found = []
    for t in TABLES:
        for ext in (".parquet", ".csv"):
            p = out_dir / f"{t}{ext}"
            if p.exists():
                found.append(p)
    return found


def write_dataset(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: GeneratorConfig,
                  fmt: str = "both", force: bool = False) -> list[Path]:
    """Write all tables plus ``metadata.json``. Refuses to overwrite unless ``force``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clash = existing_outputs(out_dir)
    if clash and not force:
        raise FileExistsError(
            "Dataset files already exist (e.g. " + clash[0].name + "). "
            "Re-run with --force to overwrite them explicitly."
        )
    written = []
    for name, df in tables.items():
        if fmt in ("parquet", "both"):
            p = out_dir / f"{name}.parquet"
            df.to_parquet(p, index=False)
            written.append(p)
        if fmt in ("csv", "both"):
            p = out_dir / f"{name}.csv"
            df.to_csv(p, index=False)
            written.append(p)
        if fmt == "parquet":  # avoid stale CSV mirrors from a previous run
            (out_dir / f"{name}.csv").unlink(missing_ok=True)
        if fmt == "csv":
            (out_dir / f"{name}.parquet").unlink(missing_ok=True)
    meta = {
        "generator_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": asdict(cfg),
        "row_counts": {k: int(len(v)) for k, v in tables.items()},
        "notes": "All records are synthetic. Not affiliated with any manufacturer or dealership.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    written.append(out_dir / "metadata.json")
    return written
