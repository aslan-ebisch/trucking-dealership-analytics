"""Static business model for the fictional dealership group.

This module holds the *assumptions* behind the synthetic data: which stores
exist, which franchises they hold, what each product costs, and a stylized
market cycle. Keeping these in one place makes the generator readable and lets
a reviewer challenge any assumption without digging through sampling code.

All dealership names, people, customers and transactions are fictional.
Manufacturer and model names are used only as ordinary product descriptors;
no affiliation with any manufacturer is implied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dealership network
# ---------------------------------------------------------------------------
# volume_weight  relative sales throughput (Dallas = 1.00 flagship)
# franchises     new-truck brands the store is franchised to sell
# trailer_lines  new-trailer brands carried
# price_factor   local pricing power (multiplies selling price)
# margin_adj     local competitive intensity, in margin points (0.005 = +0.5 pt)
# segment_tilt   multipliers on customer-segment demand around the store
DEALERSHIPS = [
    dict(dealership_id="D01", dealership_name="Trinity Truck Center", city="Dallas", state="TX",
         region="North Texas", latitude=32.7767, longitude=-96.7970, opened_date="2009-04-01",
         volume_weight=1.00, franchises=["Freightliner", "Western Star"], trailer_lines=["Utility", "Fontaine"],
         price_factor=1.00, margin_adj=-0.002, lot_capacity=420,
         segment_tilt={"National Fleet": 1.3, "Regional Carrier": 1.2}),
    dict(dealership_id="D02", dealership_name="Gulf Coast Truck Center", city="Houston", state="TX",
         region="Gulf Coast", latitude=29.7604, longitude=-95.3698, opened_date="2011-09-01",
         volume_weight=0.92, franchises=["Peterbilt"], trailer_lines=["Great Dane", "Fontaine"],
         price_factor=0.99, margin_adj=-0.004, lot_capacity=380,
         segment_tilt={"Construction & Vocational": 1.5, "Regional Carrier": 1.1}),
    dict(dealership_id="D03", dealership_name="Red Dirt Truck Center", city="Oklahoma City", state="OK",
         region="Oklahoma", latitude=35.4676, longitude=-97.5164, opened_date="2013-03-01",
         volume_weight=0.60, franchises=["Kenworth"], trailer_lines=["Wabash"],
         price_factor=1.00, margin_adj=0.002, lot_capacity=240,
         segment_tilt={"Construction & Vocational": 1.3, "Agriculture": 1.2}),
    dict(dealership_id="D04", dealership_name="Green Country Truck Center", city="Tulsa", state="OK",
         region="Oklahoma", latitude=36.1540, longitude=-95.9928, opened_date="2015-06-01",
         volume_weight=0.46, franchises=["Volvo", "Mack"], trailer_lines=["Utility"],
         price_factor=1.00, margin_adj=0.003, lot_capacity=190,
         segment_tilt={"Small Fleet": 1.2}),
    dict(dealership_id="D05", dealership_name="Heartland Truck Center", city="Kansas City", state="MO",
         region="Missouri", latitude=39.0997, longitude=-94.5786, opened_date="2012-01-01",
         volume_weight=0.68, franchises=["Peterbilt"], trailer_lines=["Great Dane", "Wabash"],
         price_factor=1.01, margin_adj=0.000, lot_capacity=270,
         segment_tilt={"Regional Carrier": 1.3, "National Fleet": 1.1}),
    dict(dealership_id="D06", dealership_name="Ozark Gateway Truck Center", city="Springfield", state="MO",
         region="Missouri", latitude=37.2089, longitude=-93.2923, opened_date="2016-08-01",
         volume_weight=0.27, franchises=["Kenworth"], trailer_lines=["Wabash"],
         price_factor=0.99, margin_adj=0.006, lot_capacity=110,
         segment_tilt={"Owner-Operator": 1.4, "Agriculture": 1.4}),
    dict(dealership_id="D07", dealership_name="Arkansas River Truck Center", city="Little Rock", state="AR",
         region="Arkansas", latitude=34.7465, longitude=-92.2896, opened_date="2014-05-01",
         volume_weight=0.40, franchises=["Freightliner", "Western Star"], trailer_lines=["Utility"],
         price_factor=0.98, margin_adj=0.004, lot_capacity=160,
         segment_tilt={"Agriculture": 1.5, "Government & Municipal": 1.2}),
    dict(dealership_id="D08", dealership_name="Sugar Creek Truck Center", city="Springdale", state="AR",
         region="Northwest Arkansas", latitude=36.1867, longitude=-94.1288, opened_date="2018-02-01",
         volume_weight=0.31, franchises=["Volvo", "Mack"], trailer_lines=["Great Dane"],
         price_factor=0.99, margin_adj=-0.001, lot_capacity=130,
         segment_tilt={"National Fleet": 2.0, "Regional Carrier": 1.3}),
    # Acquired mid-2022 and opened in 2023 — real dealership groups grow by acquisition,
    # and these late starts create genuine "new store" growth in the history.
    dict(dealership_id="D09", dealership_name="Front Range Truck Center", city="Denver", state="CO",
         region="Rocky Mountain", latitude=39.7392, longitude=-104.9903, opened_date="2022-04-01",
         volume_weight=0.55, franchises=["Kenworth"], trailer_lines=["Utility", "Wabash"],
         price_factor=1.035, margin_adj=0.003, lot_capacity=220,
         segment_tilt={"Construction & Vocational": 1.5, "Government & Municipal": 1.2}),
    dict(dealership_id="D10", dealership_name="Music City Truck Center", city="Nashville", state="TN",
         region="Middle Tennessee", latitude=36.1627, longitude=-86.7816, opened_date="2023-03-01",
         volume_weight=0.58, franchises=["Peterbilt"], trailer_lines=["Great Dane", "Utility"],
         price_factor=1.01, margin_adj=0.001, lot_capacity=230,
         segment_tilt={"Regional Carrier": 1.4, "Small Fleet": 1.1}),
]

# Nearby states used for customer home locations around each store.
STORE_CUSTOMER_STATES = {
    "D01": ["TX", "TX", "TX", "OK"], "D02": ["TX", "TX", "TX", "LA"], "D03": ["OK", "OK", "OK", "TX", "KS"],
    "D04": ["OK", "OK", "AR", "KS"], "D05": ["MO", "MO", "KS", "KS", "IA"], "D06": ["MO", "MO", "MO", "AR"],
    "D07": ["AR", "AR", "AR", "MS", "LA"], "D08": ["AR", "AR", "MO", "OK"], "D09": ["CO", "CO", "CO", "WY", "NM"],
    "D10": ["TN", "TN", "TN", "KY", "AL"],
}
NATIONAL_FLEET_HQ_STATES = ["TX", "AR", "TN", "MO", "IL", "GA", "OH", "IN", "WI", "CA", "UT", "NE"]

# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
TRUCK_CATEGORIES = [
    "Sleeper Tractor", "Day Cab Tractor", "Vocational Truck", "Dump Truck", "Flatbed Truck", "Box Truck",
]
TRAILER_CATEGORIES = ["Dry Van Trailer", "Refrigerated Trailer", "Flatbed Trailer"]
VEHICLE_CATEGORIES = TRUCK_CATEGORIES + TRAILER_CATEGORIES

# Typical annual miles by category (used-truck odometer model). Trailers: not tracked.
ANNUAL_MILES = {
    "Sleeper Tractor": 105_000, "Day Cab Tractor": 68_000, "Vocational Truck": 24_000,
    "Dump Truck": 21_000, "Flatbed Truck": 32_000, "Box Truck": 29_000,
}

# (manufacturer, model, vehicle_category, base new price in Jan-2021 dollars, popularity weight)
_CATALOG_ROWS = [
    ("Peterbilt", "579", "Sleeper Tractor", 162_000, 1.00),
    ("Peterbilt", "579 Day Cab", "Day Cab Tractor", 141_000, 0.60),
    ("Peterbilt", "589", "Sleeper Tractor", 182_000, 0.40),
    ("Peterbilt", "567", "Vocational Truck", 166_000, 0.50),
    ("Peterbilt", "567 Dump", "Dump Truck", 178_000, 0.40),
    ("Peterbilt", "548", "Flatbed Truck", 114_000, 0.25),
    ("Peterbilt", "536", "Box Truck", 99_000, 0.40),
    ("Kenworth", "T680", "Sleeper Tractor", 163_000, 1.00),
    ("Kenworth", "T680 Day Cab", "Day Cab Tractor", 142_000, 0.60),
    ("Kenworth", "W990", "Sleeper Tractor", 184_000, 0.40),
    ("Kenworth", "T880", "Vocational Truck", 168_000, 0.50),
    ("Kenworth", "T880 Dump", "Dump Truck", 180_000, 0.40),
    ("Kenworth", "T380", "Flatbed Truck", 112_000, 0.25),
    ("Kenworth", "T280", "Box Truck", 97_000, 0.40),
    ("Freightliner", "Cascadia", "Sleeper Tractor", 150_000, 1.25),
    ("Freightliner", "Cascadia Day Cab", "Day Cab Tractor", 129_000, 1.00),
    ("Freightliner", "122SD", "Vocational Truck", 162_000, 0.40),
    ("Freightliner", "114SD Dump", "Dump Truck", 168_000, 0.45),
    ("Freightliner", "M2 106 Flatbed", "Flatbed Truck", 99_000, 0.35),
    ("Freightliner", "M2 106", "Box Truck", 93_000, 0.75),
    ("Volvo", "VNL 860", "Sleeper Tractor", 158_000, 1.00),
    ("Volvo", "VNR", "Day Cab Tractor", 134_000, 0.80),
    ("Volvo", "VHD", "Vocational Truck", 161_000, 0.40),
    ("Volvo", "VHD Dump", "Dump Truck", 170_000, 0.25),
    ("Mack", "Anthem", "Sleeper Tractor", 152_000, 0.70),
    ("Mack", "Anthem Day Cab", "Day Cab Tractor", 131_000, 0.60),
    ("Mack", "Granite", "Vocational Truck", 160_000, 0.50),
    ("Mack", "Granite Dump", "Dump Truck", 172_000, 0.60),
    ("Mack", "MD7 Flatbed", "Flatbed Truck", 95_000, 0.20),
    ("Mack", "MD7", "Box Truck", 91_000, 0.30),
    ("Western Star", "57X", "Sleeper Tractor", 168_000, 0.50),
    ("Western Star", "47X", "Day Cab Tractor", 142_000, 0.40),
    ("Western Star", "49X", "Vocational Truck", 172_000, 0.60),
    ("Western Star", "49X Dump", "Dump Truck", 180_000, 0.40),
    ("Utility", "4000D-X Composite", "Dry Van Trailer", 43_000, 1.00),
    ("Utility", "3000R", "Refrigerated Trailer", 63_000, 0.80),
    ("Utility", "4000AE Flatbed", "Flatbed Trailer", 39_000, 0.50),
    ("Great Dane", "Champion CL", "Dry Van Trailer", 42_000, 1.00),
    ("Great Dane", "Everest", "Refrigerated Trailer", 65_000, 0.80),
    ("Great Dane", "Freedom LT", "Flatbed Trailer", 40_000, 0.40),
    ("Wabash", "DuraPlate", "Dry Van Trailer", 41_000, 1.00),
    ("Wabash", "ArcticLite", "Refrigerated Trailer", 61_000, 0.60),
    ("Fontaine", "Revolution", "Flatbed Trailer", 45_000, 0.60),
    ("Fontaine", "Velocity", "Flatbed Trailer", 43_000, 0.40),
]


def product_catalog() -> pd.DataFrame:
    """Return the product catalog with derived vehicle_type and category weights."""
    cat = pd.DataFrame(
        _CATALOG_ROWS,
        columns=["manufacturer", "model", "vehicle_category", "base_price", "popularity"],
    )
    cat["vehicle_type"] = np.where(cat["vehicle_category"].isin(TRAILER_CATEGORIES), "Trailer", "Truck")
    # Relative demand for each category (combined with model popularity when sampling).
    category_demand = {
        "Sleeper Tractor": 1.00, "Day Cab Tractor": 0.80, "Vocational Truck": 0.55, "Dump Truck": 0.45,
        "Flatbed Truck": 0.30, "Box Truck": 0.55,
        "Dry Van Trailer": 1.00, "Refrigerated Trailer": 0.70, "Flatbed Trailer": 0.55,
    }
    cat["sample_weight"] = cat["popularity"] * cat["vehicle_category"].map(category_demand)
    return cat


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
CUSTOMER_SEGMENTS = [
    "Owner-Operator", "Small Fleet", "Regional Carrier", "National Fleet",
    "Construction & Vocational", "Agriculture", "Government & Municipal",
]
# share of the customer *base* (not of sales)
SEGMENT_CUSTOMER_SHARE = [0.50, 0.25, 0.075, 0.015, 0.08, 0.055, 0.025]
# typical fleet size ranges (power units) per segment
SEGMENT_FLEET_SIZE = {
    "Owner-Operator": (1, 2), "Small Fleet": (2, 10), "Regional Carrier": (11, 180),
    "National Fleet": (400, 6000), "Construction & Vocational": (4, 90), "Agriculture": (1, 30),
    "Government & Municipal": (10, 350),
}
# purchase propensity: log-normal median per customer — drives repeat buying / concentration
SEGMENT_PROPENSITY = {
    "Owner-Operator": 1.0, "Small Fleet": 2.6, "Regional Carrier": 9.0, "National Fleet": 38.0,
    "Construction & Vocational": 5.0, "Agriculture": 2.0, "Government & Municipal": 5.5,
}

# P(segment | vehicle category); rows need not sum to 1 (normalized at use).
CATEGORY_SEGMENT_MIX = {
    #                      OO    SF    RC    NF    Con   Ag    Gov
    "Sleeper Tractor":     [.20, .25, .22, .20, .00, .02, .01],
    "Day Cab Tractor":     [.08, .22, .30, .25, .06, .04, .01],
    "Vocational Truck":    [.03, .10, .04, .02, .50, .10, .20],
    "Dump Truck":          [.09, .10, .00, .00, .55, .05, .18],
    "Flatbed Truck":       [.07, .25, .08, .02, .30, .15, .10],
    "Box Truck":           [.10, .35, .15, .10, .05, .05, .15],
    "Dry Van Trailer":     [.06, .25, .30, .33, .00, .02, .00],
    "Refrigerated Trailer": [.05, .20, .35, .30, .00, .07, .00],
    "Flatbed Trailer":     [.10, .25, .20, .08, .20, .10, .02],
}

# F&I behaviour by segment: P(dealer-arranged financing), P(protection product)
SEGMENT_FINANCE_RATE = {
    "Owner-Operator": 0.72, "Small Fleet": 0.55, "Regional Carrier": 0.32, "National Fleet": 0.08,
    "Construction & Vocational": 0.45, "Agriculture": 0.52, "Government & Municipal": 0.00,
}
SEGMENT_PROTECTION_RATE = {
    "Owner-Operator": 0.62, "Small Fleet": 0.50, "Regional Carrier": 0.34, "National Fleet": 0.12,
    "Construction & Vocational": 0.40, "Agriculture": 0.45, "Government & Municipal": 0.18,
}
TRUCK_PROTECTION_PRODUCTS = [
    "Extended Powertrain Warranty", "Aftertreatment Coverage", "Comprehensive Service Contract",
    "GAP Protection",
]
TRAILER_PROTECTION_PRODUCTS = ["Trailer Protection Plan"]

# Sale channel probabilities by segment.
SALE_CHANNELS = ["Showroom / Walk-In", "Phone / Referral", "Online Lead", "Fleet Account", "Government Bid"]
SEGMENT_CHANNEL_MIX = {
    "Owner-Operator":            [.52, .28, .20, .00, .00],
    "Small Fleet":               [.38, .32, .18, .12, .00],
    "Regional Carrier":          [.12, .22, .08, .58, .00],
    "National Fleet":            [.00, .08, .02, .90, .00],
    "Construction & Vocational": [.35, .35, .10, .20, .00],
    "Agriculture":               [.55, .30, .15, .00, .00],
    "Government & Municipal":    [.00, .05, .00, .05, .90],
}

ACQUISITION_SOURCES_USED = ["Trade-In", "Fleet Buyback", "Auction", "Dealer Trade"]
ACQUISITION_SOURCES_USED_P = [0.46, 0.28, 0.20, 0.06]

# ---------------------------------------------------------------------------
# Stylized market cycle (NOT actual industry data)
# ---------------------------------------------------------------------------
# A simplified freight-cycle narrative: a 2021–22 boom with constrained new-truck
# supply and record used values, a 2023–24 freight recession with collapsing used
# prices, and a 2025–26 recovery with pre-buying ahead of new emissions rules.
# Anchor points are linearly interpolated by day.
_MARKET_ANCHORS = pd.DataFrame(
    [
        # date,       volume, new_price, used_value, new_supply_tight, floorplan_rate
        ("2019-01-01", 0.95, 0.93, 0.95, 0.00, 0.052),
        ("2020-06-01", 0.80, 0.97, 0.90, 0.00, 0.035),
        ("2021-01-01", 0.94, 1.00, 1.05, 0.40, 0.032),
        ("2021-09-01", 1.06, 1.06, 1.38, 1.00, 0.032),
        ("2022-03-01", 1.08, 1.11, 1.52, 1.00, 0.038),
        ("2022-10-01", 1.04, 1.16, 1.36, 0.80, 0.062),
        ("2023-04-01", 0.95, 1.21, 1.08, 0.30, 0.080),
        ("2023-10-01", 0.88, 1.23, 0.94, -0.30, 0.085),
        ("2024-06-01", 0.86, 1.25, 0.86, -0.60, 0.085),
        ("2024-12-01", 0.90, 1.27, 0.88, -0.40, 0.080),
        ("2025-06-01", 0.95, 1.30, 0.92, -0.10, 0.075),
        ("2025-12-01", 1.00, 1.33, 0.96, 0.20, 0.072),
        ("2026-06-01", 1.08, 1.37, 1.00, 0.60, 0.070),
        ("2026-12-31", 1.12, 1.40, 1.02, 0.80, 0.070),
    ],
    columns=["date", "volume", "new_price", "used_value", "new_supply_tight", "floorplan_rate"],
)
_MARKET_ANCHORS["date"] = pd.to_datetime(_MARKET_ANCHORS["date"])

# Month-of-year seasonality for unit volume (Dec = year-end tax-motivated buying).
MONTH_SEASONALITY = np.array([0.84, 0.90, 1.05, 1.00, 1.02, 1.08, 0.94, 0.98, 1.05, 1.01, 0.95, 1.18])
# Day-of-week factor (Mon..Sun): commercial dealers close little business on Sundays.
WEEKDAY_FACTOR = np.array([1.05, 1.10, 1.10, 1.08, 1.15, 0.45, 0.07])


def market_index(dates, column: str) -> np.ndarray:
    """Interpolate a market-cycle series (see ``_MARKET_ANCHORS``) at the given dates."""
    d = pd.to_datetime(pd.Series(np.asarray(dates).ravel()))
    x = d.values.astype("datetime64[D]").astype(np.int64)
    xp = _MARKET_ANCHORS["date"].values.astype("datetime64[D]").astype(np.int64)
    return np.interp(x, xp, _MARKET_ANCHORS[column].to_numpy())


# ---------------------------------------------------------------------------
# Name fragments for fictional entities (deterministically combined by the generator)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "James", "Maria", "Robert", "Linda", "Michael", "Angela", "David", "Teresa", "Carlos", "Jennifer",
    "Brian", "Monica", "Kevin", "Rachel", "Jason", "Denise", "Marcus", "Kelly", "Travis", "Erica",
    "Derek", "Brandy", "Luis", "Tamara", "Wesley", "Crystal", "Cody", "Heather", "Dustin", "Rosa",
    "Shane", "Amber", "Tyrone", "Leah", "Garrett", "Nicole", "Hector", "Paige", "Clint", "Vanessa",
    "Darnell", "Kristen", "Ramon", "Megan", "Colton", "Jasmine", "Trent", "Alicia", "Wade", "Gloria",
]
LAST_NAMES = [
    "Whitaker", "Delgado", "Holloway", "Pruitt", "Castillo", "Brennan", "McAllister", "Okafor", "Ramsey",
    "Vance", "Thornton", "Guerrero", "Lockhart", "Pennington", "Salazar", "Hargrove", "Ellison", "Barlow",
    "Navarro", "Kincaid", "Sutton", "Mercer", "Dawson", "Fairchild", "Montoya", "Calloway", "Wakefield",
    "Rhodes", "Bledsoe", "Ochoa", "Stanfield", "Carver", "Lindqvist", "Atwood", "Beckett", "Crenshaw",
    "Duvall", "Escobar", "Garland", "Haskell", "Ingram", "Jessup", "Kendrick", "Langley", "Maddox",
    "Norwood", "Overstreet", "Pickett", "Quintero", "Redford", "Stroud", "Tolliver", "Upchurch", "Voss",
    "Whitfield", "Yarbrough", "Zamora", "Ashby", "Burkett", "Colquitt",
]
COMPANY_PREFIXES = [
    "Red River", "Iron Horse", "Blue Mesa", "Prairie Wind", "Lone Oak", "Cedar Creek", "Black Mesa",
    "Silver Spur", "High Plains", "Big Sky", "Cross Timbers", "Twin Rivers", "Granite Ridge", "Copper Canyon",
    "Pine Hollow", "Stone Bridge", "Eagle Rock", "Bluestem", "White Oak", "Southern Cross", "Arrowhead",
    "Sandstone", "Thunder Ridge", "Long Haul", "Midland Star", "Ozark Ridge", "Cimarron", "Brazos Valley",
    "Pecos", "Sabine", "Neosho", "Wildhorse", "Red Rock", "Clearwater", "Frontier", "Heartland Ridge",
    "Gold Prairie", "Bluebonnet", "Ironwood", "Kingfisher", "Painted Sky", "Riverbend", "Sawtooth",
    "Timberline", "Westbrook", "Yellowstone Trail", "Summit Peak", "Canyon Creek", "Shady Grove",
    "Rolling Hills", "Mustang", "Stockyard", "Northfork", "Lakeside", "Old Mill", "Trailhead",
    "Hickory Lane", "Magnolia", "Persimmon", "Quarry Hill",
]
COMPANY_CORES = {
    "Owner-Operator": ["Trucking", "Hauling", "Transport", "Express"],
    "Small Fleet": ["Freight", "Transport", "Logistics", "Carriers", "Hauling"],
    "Regional Carrier": ["Freight Lines", "Logistics", "Transportation", "Express", "Carriers"],
    "National Fleet": ["Freightways", "Logistics Group", "Transportation Inc.", "Carriers National"],
    "Construction & Vocational": ["Construction", "Excavation", "Paving", "Aggregates", "Materials",
                                  "Earthworks"],
    "Agriculture": ["Farms", "Grain Co.", "Cattle Co.", "Ag Services", "Feed & Grain"],
    "Government & Municipal": ["Public Works District", "Road & Bridge District", "Water Utility Authority",
                               "Solid Waste Authority", "Transit District"],
}
COMPANY_SUFFIXES = ["LLC", "Inc.", "Co.", "LLC", "LLC", "Group"]
