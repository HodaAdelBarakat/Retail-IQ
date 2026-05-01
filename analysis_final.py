import pandas as pd
import numpy as np
from utils_ import load_sales_file, safe_divide

# ============================================================
# الإعدادات العامة (Configuration)
# ============================================================
VAT_RATE = 0.14          # 14% ضريبة قيمة مضافة
INCOME_TAX_RATE = 0.225  # 22.5% ضريبة دخل على الأرباح

def load_and_clean_data(df: pd.DataFrame):
    """
    The Master Data Cleaning Pipeline (Ultra-Robust)
    تدمج كافة استراتيجيات التنظيف، وتدعم Online Retail وكافة مجموعات بيانات Kaggle.
    """
    
    # 1. 🚀 المرحلة الأولى: التوحيد الشامل (Normalization)
    # تحويل كل أسماء الأعمدة لحروف صغيرة، إزالة المسافات، واستبدال الشرطات (_) والشرطات (-) بمسافات
    df.columns = [str(col).strip().lower().replace('_', ' ').replace('-', ' ') for col in df.columns]
    
    # 2. 🧠 المرحلة الثانية: القاموس الخارق الموحد (The Mega Dictionary)
    # جميع المفاتيح هنا مكتوبة بحروف صغيرة لتتطابق مع عملية التوحيد أعلاه
    column_aliases = {
        # تواريخ
        'invoicedate': 'Order Date', 'date': 'Order Date', 'orderdate': 'Order Date',
        'order date': 'Order Date', 'date/time': 'Order Date', 'invoice date': 'Order Date',
        'order date (dateorders)': 'Order Date', 'shipping date (dateorders)': 'Ship Date',
        
        # مبيعات وأسعار
        'revenue': 'Sales', 'total': 'Sales', 'total price': 'Sales', 'sales per customer': 'Sales', 
        'sales': 'Sales', 'price': 'UnitPrice', 'unit price': 'UnitPrice', 
        'unitprice': 'UnitPrice', 'product price': 'UnitPrice', 'item price': 'UnitPrice',
        
        # أرباح وخصومات
        'margin': 'Profit', 'order profit per order': 'Profit', 'profit per order': 'Profit', 
        'profit': 'Profit', 'discount': 'Discount', 'discount amount': 'Discount',
        
        # عملاء
        'customer id': 'Customer ID', 'customerid': 'Customer ID', 'customer': 'Customer Name', 
        'customer first name': 'Customer Name', 'customer name': 'Customer Name',
        
        # كميات ومعرفات وفئات
        'qty': 'Quantity', 'order item quantity': 'Quantity', 'quantity': 'Quantity',
        'invoiceno': 'Order ID', 'order id': 'Order ID', 'invoice no': 'Order ID',
        'product name': 'Product Name', 'product description': 'Product Name',
        'description': 'Product Name', # دعم خاص لـ Online Retail
        'category name': 'Category', 'product category': 'Category', 'department name': 'Category', 
        'category': 'Category'
    }
    
    # تطبيق الترجمة بناءً على القاموس
    df.rename(columns=column_aliases, inplace=True)

    # 🛡️ الحماية القصوى: التخلص من أي أعمدة مكررة نتجت عن عملية الترجمة
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 3. 🎯 المرحلة الثالثة: التوليد الذاتي للبيانات الناقصة (Synthetic Logic)
    # إذا كانت المبيعات مفقودة ولكن لدينا الكمية والسعر -> نحسب المبيعات
    if 'Sales' not in df.columns and 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Sales'] = pd.to_numeric(df['Quantity'], errors='coerce') * pd.to_numeric(df['UnitPrice'], errors='coerce')
        
    # إذا كان الربح مفقوداً -> نفترض هامش ربح 15% كقيمة افتراضية لضمان عمل التحليلات
    if 'Profit' not in df.columns and 'Sales' in df.columns:
        df['Profit'] = pd.to_numeric(df['Sales'], errors='coerce') * 0.15

    # إذا كانت الكمية مفقودة -> نفترض أن كل سطر يمثل قطعة واحدة
    if 'Quantity' not in df.columns:
        df['Quantity'] = 1

    # 4. ⚠️ المرحلة الرابعة: التحقق من "أعمدة الحياة" (Validation)
    required_columns = ['Order Date', 'Sales', 'Profit']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return None, f"⚠️ Data Format Error: This file does not look like financial sales data. Missing: {', '.join(missing_columns)}"

    # تنظيف التواريخ والأرقام
    df = df.dropna(subset=['Order Date', 'Sales'])
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df.dropna(subset=['Order Date'])
    
    # تحويل كافة الأعمدة المالية إلى أرقام حقيقية ومعالجة القيم الفارغة بـ 0
    for col in ['Sales', 'Profit', 'Discount', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if 'Discount' not in df.columns:
        df['Discount'] = 0.0

    # استبعاد الطلبات ذات الكمية صفر أو سالبة (بيانات خاطئة)
    df = df[df["Quantity"] > 0].copy()

    # 5. 🛠️ المرحلة الخامسة: سد ثغرات الهوية (Identity Patching)
    # توليد معرف عميل من اسمه إذا كان مفقوداً
    if "Customer ID" not in df.columns and "Customer Name" in df.columns:
        normalized_names = df["Customer Name"].astype(str).str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
        df["Customer ID"] = "CUST_" + normalized_names
    elif "Customer Name" not in df.columns:
        df["Customer ID"] = "UNKNOWN"
        df["Customer Name"] = "Walk-in Customer"

    # توليد أرقام فواتير تسلسلية إذا كانت مفقودة
    if 'Order ID' not in df.columns:
        df['Order ID'] = ['ORD-' + str(i) for i in range(1, len(df) + 1)]
        
    # تعيين فئة عامة إذا كانت مفقودة
    if 'Category' not in df.columns:
        df['Category'] = "General Retail"

    # 6. 💰 المرحلة السادسة: الهندسة المالية والضريبية
    df["COGS"] = df["Sales"] - df["Profit"]
    df["Cost per Unit"] = df["COGS"] / df["Quantity"]

    VAT_RATE = 0.14
    INCOME_TAX_RATE = 0.225
    df["VAT_Amount"]        = df["Sales"] * VAT_RATE
    df["Sales_After_VAT"]   = df["Sales"] + df["VAT_Amount"]
    df["Income_Tax"]        = df["Profit"].clip(lower=0) * INCOME_TAX_RATE
    df["Net_Profit_AfterTax"] = df["Profit"] - df["Income_Tax"]

    # تحديد الفواتير المشبوهة ضريبياً (ربح صفري/سالب مع خصم > 20%)
    df["Tax_Suspicious"] = (df["Profit"] <= 0) & (df["Discount"] > 0.20)

    return df.reset_index(drop=True), None

def calculate_kpis(df: pd.DataFrame) -> dict:
    """حساب مؤشرات الأداء المالية"""
    total_sales  = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Order ID"].nunique()) if "Order ID" in df.columns else len(df)
    total_cost = total_sales - total_profit
    cost_to_sales_ratio = safe_divide(total_cost, total_sales) * 100

    if "Order Date" in df.columns:
        date_range_days = max((df["Order Date"].max() - df["Order Date"].min()).days, 1)
        order_velocity  = round(safe_divide(total_orders, date_range_days), 2)
    else:
        order_velocity = 0.0

    kpis = {
        "Total Sales":          total_sales,
        "Total Profit":         total_profit,
        "Total Cost (COGS)":    total_cost, 
        "Cost to Sales Ratio":  cost_to_sales_ratio,
        "Profit Margin":        safe_divide(total_profit, total_sales),
        "Average Order Value":  safe_divide(total_sales, total_orders),
        "Total Orders":         total_orders,
        "Order Velocity":       order_velocity,
        "Total VAT":            float(df["VAT_Amount"].sum()) if "VAT_Amount" in df.columns else 0.0,
        "Total Income Tax":     float(df["Income_Tax"].sum()) if "Income_Tax" in df.columns else 0.0,
        "Net Profit After Tax": float(df["Net_Profit_AfterTax"].sum()) if "Net_Profit_AfterTax" in df.columns else 0.0,
        "Tax Suspicious Count": int(df["Tax_Suspicious"].sum()) if "Tax_Suspicious" in df.columns else 0,
    }
    return kpis

def category_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """تحليل أداء الفئات"""
    if "Category" not in df.columns:
        df["Category"] = "General" # توليد فئة افتراضية لو مش موجودة
    return df.groupby("Category", as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Average_Discount=("Discount", "mean"),
    ).sort_values(by="Total_Sales", ascending=False)

def product_analysis(df: pd.DataFrame, top_n: int = 10) -> tuple:
    """تحليل المنتجات: الرابحة والنازفة"""
    if "Product Name" not in df.columns:
        df["Product Name"] = "Item " + df["Order ID"].astype(str)
        
    summary = df.groupby("Product Name", as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Average_Discount=("Discount", "mean"),
    ).sort_values(by="Total_Profit", ascending=False)

    top_products   = summary.head(top_n).reset_index(drop=True)
    worst_products = summary.sort_values(by="Total_Profit", ascending=True).head(top_n).reset_index(drop=True)
    return top_products, worst_products

def customer_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """تحليل Pareto 80/20 لتحديد العملاء الاستراتيجيين VIP"""
    if "Customer Name" not in df.columns:
        return pd.DataFrame()

    customer_sales = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
    ).sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    customer_sales["CumSum"] = customer_sales["Total_Sales"].cumsum()
    total_rev = customer_sales["Total_Sales"].sum()
    customer_sales["CumPerc"] = customer_sales["CumSum"] / total_rev

    vip_customers = customer_sales[customer_sales["CumPerc"] <= 0.80].copy()
    vip_customers["VIP Segment"] = "Pareto 80% Revenue"
    
    return vip_customers.drop(columns=["CumSum", "CumPerc"])

def get_tax_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """إنشاء جدول التدقيق الضريبي التفصيلي"""
    cols_wanted = [
        "Order ID", "Order Date", "Customer Name", "Product Name", "Category",
        "Sales", "Profit", "Discount", "VAT_Amount", "Sales_After_VAT",
        "Income_Tax", "Net_Profit_AfterTax", "Tax_Suspicious"
    ]
    available = [c for c in cols_wanted if c in df.columns]
    tax_df = df[available].copy()

    rename_map = {
        "VAT_Amount":           "VAT (14%)",
        "Sales_After_VAT":      "Total Incl. VAT",
        "Income_Tax":           "Income Tax (22.5%)",
        "Net_Profit_AfterTax":  "Net Profit After Tax",
        "Tax_Suspicious":       "Tax Suspicious ⚠️",
    }
    tax_df.rename(columns=rename_map, inplace=True)

    if "Tax Suspicious ⚠️" in tax_df.columns:
        tax_df["Tax Suspicious ⚠️"] = tax_df["Tax Suspicious ⚠️"].map({True: "⚠️ Yes", False: "✅ No"})

    return tax_df

def build_analysis_bundle(file_buffer) -> dict:
    """تجميع كافة التحليلات في حزمة واحدة لإرسالها للواجهة ومحرك AI"""
    # 1. قراءة الملف بشكل صحيح عبر دالة utils_
    raw_df = load_sales_file(file_buffer)
    
    # 2. تنظيف البيانات (هنا نستقبل الداتا فريم والخطأ إن وُجد)
    df, error = load_and_clean_data(raw_df)
    
    # إذا كان هناك خطأ لا يمكن تجاوزه، نعيده للواجهة
    if error:
        return {"error": error}
        
    top_products, worst_products = product_analysis(df)

    return {
        "cleaned_data":     df,
        "kpis":             calculate_kpis(df),
        "category_summary": category_analysis(df),
        "top_products":     top_products,
        "worst_products":   worst_products,
        "vip_customers":    customer_analysis(df),
        "tax_audit_table":  get_tax_audit_table(df),
    }
