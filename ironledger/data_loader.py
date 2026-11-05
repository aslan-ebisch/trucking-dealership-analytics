"""Data-access layer.

The rest of the application asks for a ``Dataset`` and never touches files
directly. Today the only backend is ``LocalFileSource`` (Parquet, falling back
to CSV). A future ``AzureBlobSource`` or ``AzureSqlSource`` only needs to
implement the three methods of ``DataSource`` — analytics and UI code stay
unchanged.

This module has no Streamlit dependency (caching lives in ``ui/data.py``), so it
can be reused by tests, notebooks, or batch jobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from ironledger import config

DATE_COLUMNS = {
    "dealerships": ["opened_date"],
    "salespeople": ["hire_date", "termination_date"],
    "customers": ["customer_since"],
    "inventory": ["acquisition_date", "sale_date"],
    "sales": ["sale_date", "inventory_acquisition_date"],
    "operating_expenses": ["month"],
}

# Low-cardinality text columns stored as pandas Categoricals: ~5-10x less memory
# and faster group-bys, which matters as the dataset scales to 100k+ rows.
CATEGORY_COLUMNS = {
    "sales": ["dealership_id", "dealership_name", "city", "state", "customer_segment", "manufacturer", "model",
              "vehicle_category", "vehicle_type", "condition", "acquisition_source", "protection_product",
              "sale_channel", "transaction_status"],
    "inventory": ["dealership_id", "manufacturer", "model", "vehicle_category", "vehicle_type", "condition",
                  "acquisition_source", "status"],
}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the generated datasets are not available."""


class DataSource(Protocol):
    """Minimal storage interface. Implement this to add a cloud backend."""

    def read_table(self, name: str) -> pd.DataFrame: ...

    def read_metadata(self) -> dict: ...

    def signature(self) -> str:
        """A string that changes whenever the underlying data changes (used for cache keys)."""
        ...


@dataclass
class LocalFileSource:
    base_dir: Path = field(default_factory=lambda: config.DATA_DIR)

    def _path(self, name: str) -> Path:
        for ext in (".parquet", ".csv"):
            p = Path(self.base_dir) / f"{name}{ext}"
            if p.exists():
                return p
        raise DatasetNotFoundError(
            f"Dataset '{name}' not found in {self.base_dir}. Run `python generate_data.py` first."
        )

    def read_table(self, name: str) -> pd.DataFrame:
        p = self._path(name)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, parse_dates=DATE_COLUMNS.get(name, []))

    def read_metadata(self) -> dict:
        p = Path(self.base_dir) / "metadata.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def signature(self) -> str:
        parts = []
        for name in config.TABLES:
            try:
                p = self._path(name)
                parts.append(f"{p.resolve()}:{p.stat().st_mtime_ns}:{p.stat().st_size}")
            except DatasetNotFoundError:
                parts.append(f"{name}:missing")
        return "|".join(parts)

    def file_info(self) -> pd.DataFrame:
        rows = []
        for name in config.TABLES:
            try:
                p = self._path(name)
                rows.append({"table": name, "file": p.name, "format": p.suffix.lstrip(".").upper(),
                             "size_mb": round(p.stat().st_size / 1e6, 2)})
            except DatasetNotFoundError:
                rows.append({"table": name, "file": "—", "format": "missing", "size_mb": 0.0})
        return pd.DataFrame(rows)


def get_data_source() -> DataSource:
    """Factory selected by the IRONLEDGER_DATA_SOURCE environment variable."""
    if config.DATA_SOURCE == "local":
        return LocalFileSource(config.DATA_DIR)
    raise NotImplementedError(
        f"Data source '{config.DATA_SOURCE}' is not implemented yet. "
        "Planned: 'azure_blob' (Parquet in Azure Blob Storage) and 'azure_sql'."
    )


@dataclass
class Dataset:
    dealerships: pd.DataFrame
    salespeople: pd.DataFrame
    customers: pd.DataFrame
    inventory: pd.DataFrame
    sales: pd.DataFrame
    operating_expenses: pd.DataFrame
    metadata: dict

    def tables(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in config.TABLES}


def _coerce(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    for col in CATEGORY_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    if name == "sales":
        df["financing_arranged"] = df["financing_arranged"].astype(bool)
    return df


def load_dataset(source: DataSource | None = None) -> Dataset:
    """Read every table, apply types, and add convenience reporting columns.

    The returned frames must be treated as read-only by callers (the UI caches a
    single shared copy for performance).
    """
    source = source or get_data_source()
    raw = {name: _coerce(name, source.read_table(name)) for name in config.TABLES}

    dealers = raw["dealerships"]
    sales = raw["sales"]
    sales["sale_month"] = sales["sale_date"].dt.to_period("M").dt.to_timestamp()
    sp_names = raw["salespeople"].set_index("salesperson_id")["salesperson_name"]
    cust = raw["customers"].set_index("customer_id")
    sales["salesperson_name"] = sales["salesperson_id"].map(sp_names)
    sales["customer_name"] = sales["customer_id"].map(cust["customer_name"])
    sales["region"] = sales["dealership_id"].astype(str).map(dealers.set_index("dealership_id")["region"]).astype(
        "category")

    inv = raw["inventory"]
    d = dealers.set_index("dealership_id")
    inv["dealership_name"] = inv["dealership_id"].astype(str).map(d["dealership_name"]).astype("category")
    inv["city"] = inv["dealership_id"].astype(str).map(d["city"]).astype("category")
    inv["state"] = inv["dealership_id"].astype(str).map(d["state"]).astype("category")

    opex = raw["operating_expenses"]
    opex["dealership_name"] = opex["dealership_id"].map(d["dealership_name"])

    sp = raw["salespeople"]
    sp["dealership_name"] = sp["dealership_id"].map(d["dealership_name"])

    c = raw["customers"]
    c["primary_dealership_name"] = c["primary_dealership_id"].map(d["dealership_name"])

    return Dataset(metadata=source.read_metadata(), **raw)
