"""IronLedger Analytics — commercial truck dealership intelligence.

Package layout
--------------
config          Paths, reporting window, environment-driven settings.
reference_data  Static business model: dealerships, product catalog, market cycle.
generator       Synthetic data engine (writes the datasets in ``data/``).
data_loader     Data-access layer (local Parquet/CSV today; cloud storage later).
analytics       Pure pandas business and financial calculations (no UI code).
validation      Data-quality checks shared by the test suite and the Data Explorer.
ui              Streamlit presentation helpers (styling, charts, filters, KPI cards).
"""

__version__ = "1.0.0"
