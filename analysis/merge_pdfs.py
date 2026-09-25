#!/usr/bin/env python
"""Concatenate a tables PDF with one or more figure-PDF directories into a
single combined PDF: tables first, then each --figures directory's *.pdf
files (sorted), in the order given. A --figures value may also be a single
PDF, such as a titled LaTeX figure book.

Usage:
  python analysis/merge_pdfs.py --tables path/to/summary_tables.pdf \
      --figures path/to/layer_dir path/to/paper_dir \
      --out path/to/summary_tables_with_figures.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pypdf import PdfWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.tables.is_file():
        print(f"error: tables PDF not found: {args.tables}", file=sys.stderr)
        sys.exit(1)

    writer = PdfWriter()
    writer.append(str(args.tables))
    n_figs = 0
    for fig_dir in args.figures:
        if fig_dir.is_file() and fig_dir.suffix.lower() == ".pdf":
            writer.append(str(fig_dir))
            n_figs += 1
            continue
        if not fig_dir.is_dir():
            print(f"warning: figures dir not found, skipping: {fig_dir}", file=sys.stderr)
            continue
        pdfs = sorted(fig_dir.glob("*.pdf"))
        if not pdfs:
            print(f"warning: no PDFs in {fig_dir}", file=sys.stderr)
            continue
        for pdf_path in pdfs:
            writer.append(str(pdf_path))
            n_figs += 1

    if n_figs == 0:
        print("error: no figure PDFs found to append — refusing to write a tables-only "
              "file under a '_with_figures' name (use the plain tables PDF instead)", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        writer.write(f)
    print(f"Wrote {args.out} (1 tables PDF + {n_figs} figure pages/PDFs)")


if __name__ == "__main__":
    main()
