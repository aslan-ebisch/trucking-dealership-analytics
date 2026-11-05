"""Generate the IronLedger synthetic datasets.

Usage (from the project root):

    python generate_data.py                      # default ~30k sales, Parquet + CSV
    python generate_data.py --force              # overwrite an existing dataset
    python generate_data.py --sales 100000 --customers 5000 --force   # scale test
    python generate_data.py --format parquet     # skip CSV mirrors (smaller, faster)

The dashboard never generates data itself; it only reads what this script writes.
"""

from __future__ import annotations

import argparse
import sys
import time

from ironledger.config import DATA_DIR, DEFAULT_SEED
from ironledger.generator import GeneratorConfig, existing_outputs, generate_all, write_dataset
from ironledger.validation import run_all_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate IronLedger synthetic dealership data.")
    parser.add_argument("--sales", type=int, default=30_000, help="number of sales transactions (default 30,000)")
    parser.add_argument("--customers", type=int, default=5_000, help="number of commercial customers")
    parser.add_argument("--salespeople", type=int, default=100, help="sales roster size, incl. former reps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for reproducibility")
    parser.add_argument("--out", default=str(DATA_DIR), help="output directory (default: data/)")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both",
                        help="file format(s) to write (Parquet is what the app reads first)")
    parser.add_argument("--force", action="store_true", help="overwrite existing dataset files")
    args = parser.parse_args(argv)

    if existing_outputs(args.out) and not args.force:
        print(f"Dataset already exists in '{args.out}'. Nothing was changed.\n"
              "Re-run with --force to regenerate and overwrite it.")
        return 1

    cfg = GeneratorConfig(n_sales=args.sales, n_customers=args.customers,
                          n_salespeople=args.salespeople, seed=args.seed)
    t0 = time.perf_counter()
    print(f"Generating {cfg.n_sales:,} sales | {cfg.n_customers:,} customers | "
          f"{cfg.n_salespeople} salespeople | seed {cfg.seed} ...")
    tables = generate_all(cfg)
    t_gen = time.perf_counter() - t0

    print("Validating ...")
    results = run_all_checks(tables)
    failed = [r for r in results if not r.passed]
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}" + ("" if r.passed else f" — {r.detail}"))
    if failed:
        print(f"\n{len(failed)} data-quality check(s) failed; dataset NOT written.")
        return 2

    written = write_dataset(tables, args.out, cfg, fmt=args.format, force=args.force)
    print(f"\nGenerated in {t_gen:.1f}s. Wrote {len(written)} files to '{args.out}':")
    for name, df in tables.items():
        print(f"  {name:<20} {len(df):>8,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
