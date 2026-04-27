from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any

def load_sales_file(file_path):
    if file_path.name.endswith('.csv'):
        try:
            # المحاولة الأولى: التشفير العالمي
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # المحاولة الثانية: تشفير كاجل الشهير
                file_path.seek(0)
                return pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                # المحاولة الثالثة: تشفير ويندوز القديم
                file_path.seek(0)
                return pd.read_csv(file_path, encoding='windows-1252')
    elif file_path.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
            raise ValueError("صيغة الملف غير مدعومة. يرجى رفع ملف CSV أو Excel.")

    path = Path(file_path_or_buffer)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format. Use CSV or Excel.")

def format_currency(value: float) -> str:
    """تنسيق الأرقام كعملة مع اختصار الملايين (M) والآلاف (K)"""
    if pd.isna(value) or value is None:
        return "$0.00"
        
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1_000_000_000:
        return f"{sign}${abs_val / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"{sign}${abs_val / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{sign}${abs_val / 1_000:.2f}K"
    else:
        return f"{sign}${abs_val:,.2f}"

def format_percentage(value: float) -> str:
    """تنسيق الأرقام كنسبة مئوية"""
    return f"{value:.2%}"

def format_days(value: float) -> str:
    """تنسيق الأيام"""
    return f"{value:.2f} days"

def safe_divide(numerator: float, denominator: float) -> float:
    """القسمة الآمنة لتجنب خطأ القسمة على صفر"""
    if denominator == 0:
        return 0.0
    return numerator / denominator

def to_records_table(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """تحويل الداتا فريم لقائمة من القواميس"""
    return dataframe.to_dict(orient="records")