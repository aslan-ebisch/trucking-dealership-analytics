"""Central configuration.

Everything that might differ between a laptop and a future cloud deployment
(where the data lives, which storage backend to use) is read from environment
variables here, so no other module hardcodes a path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where datasets are read from / written to. Override with IRONLEDGER_DATA_DIR.
DATA_DIR = Path(os.environ.get("IRONLEDGER_DATA_DIR", PROJECT_ROOT / "data"))

# Storage backend used by data_loader.get_data_source(). Only "local" exists today;
# "azure_blob" / "azure_sql" are the planned future implementations.
DATA_SOURCE = os.environ.get("IRONLEDGER_DATA_SOURCE", "local")

# Reporting window of the synthetic business history.
HISTORY_START = pd.Timestamp("2021-01-01")
AS_OF_DATE = pd.Timestamp("2026-10-31")  # last day with recorded activity

DEFAULT_SEED = 20260131

# Dataset names (file stems). The same names are used by every storage backend.
TABLES = (
    "dealerships",
    "salespeople",
    "customers",
    "inventory",
    "sales",
    "operating_expenses",
)

APP_NAME = "IRONLEDGER"
APP_SUBTITLE = "Commercial Truck Dealership Intelligence"
APP_TAGLINE = "Intelligence that drives the business."
GROUP_NAME = "Red River Commercial Truck Group"  # fictional dealership group
