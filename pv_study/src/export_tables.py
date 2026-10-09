# -*- coding: utf-8 -*-
"""
export_tables.py — Gather every CSV table of tables/ into one Excel workbook
(one sheet per table, formatted header, frozen first row, adjusted widths).

    python export_tables.py   → tables/Results_summary.xlsx
"""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

TABLES = Path(__file__).resolve().parents[1] / "tables"

#: order and sheet names of the workbook
SHEETS = [("best_per_configuration", "Best per configuration"), ("table_config1", "Config 1 - patch"),
          ("table_config2", "Config 2 - history"), ("table_config3", "Config 3 - channels"),
          ("table_config4", "Config 4 - horizon"), ("baseline_comparison", "Final vs baselines"),
          ("cross_validation", "Cross-validation")]


def export():
    out = TABLES / "Results_summary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for stem, sheet in SHEETS:
            f = TABLES / f"{stem}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df.to_excel(xw, sheet_name=sheet, index=False, float_format="%.4f")
            ws = xw.sheets[sheet]
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F3A5F")
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.freeze_panes = "A2"
            for j, col in enumerate(df.columns, start=1):
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(50))) + 2
                ws.column_dimensions[get_column_letter(j)].width = min(width, 40)
    print("written:", out)


if __name__ == "__main__":
    export()
