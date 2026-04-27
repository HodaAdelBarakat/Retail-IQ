from __future__ import annotations
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

def run_arima_forecast(df: pd.DataFrame):
    df_time = df.copy()
    df_time.set_index("Order Date", inplace=True)
    monthly_sales = df_time["Sales"].resample("ME").sum().fillna(0)
    
    if len(monthly_sales) < 6:
        return monthly_sales, pd.DataFrame()
        
    model = ARIMA(monthly_sales, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)
    forecast_dates = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    forecast_df = pd.DataFrame({"Forecast Sales": forecast.values}, index=forecast_dates)
    return monthly_sales, forecast_df

def get_prescriptive_directive(monthly_sales, forecast_df, anomaly_count, profit_margin):
    """توليد نصائح استراتيجية دقيقة لتجنب الخسارة أو تعظيم الربح"""
    if monthly_sales.empty or forecast_df.empty:
        return {"direction": "neutral", "actions_ar": [], "actions_en": [], "strategy_title_ar": "", "strategy_title_en": ""}
        
    last_actual = float(monthly_sales.iloc[-1])
    avg_forecast = float(forecast_df["Forecast Sales"].mean())
    trend_pct = ((avg_forecast - last_actual) / max(abs(last_actual), 1)) * 100

    if avg_forecast < last_actual or profit_margin < 0.10:
        # حالة الخسارة أو ضعف الربحية - استراتيجية تجنب الخسارة
        direction = "negative"
        title_ar = "🛡️ استراتيجية الدفاع وتجنب الخسائر (Stop-Loss)"
        title_en = "🛡️ Defensive Strategy & Loss Avoidance"
        actions_ar = [
            "⚠️ إيقاف فوري للخصومات على المنتجات التي يقل هامش ربحها عن 5%.",
            "🔍 تدقيق فوري للفواتير الحمراء (الشاذة) لاسترداد النزيف المالي.",
            "📦 تقليل حجم المخزون من الفئات الراكدة لزيادة السيولة (Cash Flow).",
            "📉 مراجعة عقود الموردين للمنتجات ذات التكلفة المرتفعة لتقليل COGS."
        ]
        actions_en = [
            "⚠️ Stop discounts on products with <5% margin immediately.",
            "🔍 Audit 'Red' anomaly invoices to recover financial leakage.",
            "📦 Reduce inventory levels for slow-moving categories to boost cash flow.",
            "📉 Renegotiate supplier contracts for high-cost items to lower COGS."
        ]
    else:
        # حالة الربحية والنمو - استراتيجية تعظيم المكاسب
        direction = "positive"
        title_ar = "🚀 استراتيجية الهجوم وتعظيم الأرباح (Profit-Max)"
        title_en = "🚀 Offensive Strategy & Profit Maximization"
        actions_ar = [
            "💰 توجيه ميزانية التسويق فوراً نحو 'كبار العملاء' (VIP) لزيادة الولاء.",
            "📈 زيادة مخزون المنتجات 'الأكثر ربحية' بنسبة 15% لمواجهة الطلب المتوقع.",
            "🎯 تفعيل سياسة الـ Upselling (عرض منتجات أعلى سعراً) للفئات الواعدة.",
            "💎 الحفاظ على مستوى الخصم الحالي لضمان الحصة السوقية مع مراقبة الهامش."
        ]
        actions_en = [
            "💰 Reallocate marketing budget to VIP customers to boost LTV.",
            "📈 Increase inventory by 15% for high-margin products.",
            "🎯 Implement Upselling strategies for promising categories.",
            "💎 Maintain current discount levels while monitoring net margins."
        ]

    return {
        "direction": direction,
        "strategy_title_ar": title_ar,
        "strategy_title_en": title_en,
        "actions_ar": actions_ar,
        "actions_en": actions_en,
        "trend_pct": f"{trend_pct:+.1f}%"
    }

def detect_anomalies(df: pd.DataFrame):
    df_anom = df.copy()
    if "Profit" in df_anom.columns:
        df_anom["profit_z"] = np.nan_to_num(stats.zscore(df_anom["Profit"].fillna(0)))
    else:
        df_anom["profit_z"] = 0
        
    if "Discount" in df_anom.columns:
        df_anom["discount_z"] = np.nan_to_num(stats.zscore(df_anom["Discount"].fillna(0)))
    else:
        df_anom["discount_z"] = 0
        
    anomalies = df_anom[(df_anom["profit_z"].abs() > 3) | (df_anom["discount_z"].abs() > 3)].copy()
    return df_anom, anomalies

def calculate_tax_risk(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة حساب مستوى الخطر الضريبي لرسم الـ Pie Chart"""
    df_tax = df.copy()
    if "Discount" not in df_tax.columns: df_tax["Discount"] = 0
    if "Profit" not in df_tax.columns: df_tax["Profit"] = 0
    
    df_tax['Tax_Risk_Score'] = (df_tax['Discount'] * 100) + (df_tax['Profit'] < 0) * 50
    df_tax['Tax_Risk_Level'] = pd.cut(df_tax['Tax_Risk_Score'], bins=[-1, 35, 65, 5000], labels=['Low', 'Medium', 'High'])
    return df_tax

def get_decisions(row) -> dict:
    """
    محرك قرارات خبير (Expert System) شامل يحاكي العقل التحليلي للمدير المالي.
    يغطي: الضرائب، النزيف، الفرص، المخزون الميت، التكاليف المفقودة، وفخ التغليف.
    """
    profit = float(row.get("Profit", 0))
    sales = float(row.get("Sales", 0))
    discount = float(row.get("Discount", 0))
    qty = float(row.get("Quantity", 1))
    profit_z = float(row.get("profit_z", 0))
    tax_suspicious = row.get("Tax_Suspicious", False)

    margin = profit / sales if sales > 0 else 0

    rec_en, rec_ar = [], []

    # 1. الامتثال الضريبي (أولوية قصوى)
    if tax_suspicious:
        rec_ar.append("⚖️ إحالة للامتثال الضريبي: تلاعب محتمل (خصم وهمي لتوليد خسارة دفترية).")
        rec_en.append("⚖️ Tax Compliance Alert: Potential manipulation (phantom discount).")

    # 2. أخطاء الإدخال القاتلة والتكاليف المفقودة (Missing COGS)
    if margin > 0.85 and sales > 100:
        rec_ar.append("🚨 ربح وهمي: الهامش يتجاوز 85%، يرجى مراجعة الفاتورة، غالباً لم يتم تسجيل تكلفة البضاعة (COGS).")
        rec_en.append("🚨 Phantom Profit: Margin > 85%. Likely missing Cost of Goods Sold (COGS) data.")
    elif profit_z < -3 and profit > 0:
        rec_ar.append("🚨 انحراف مالي حاد: مراجعة الفاتورة يدوياً لاحتمال وجود خطأ صفري (Typo) أدى لتقزيم الربح.")
        rec_en.append("🚨 Severe Financial Deviation: Review for data entry typo shrinking the profit.")

    # 3. تحليل الخسائر والنزيف المالي
    if profit < 0:
        if qty > 20 and margin > -0.05: # خسارة طفيفة مع كمية كبيرة (الطُعم)
            rec_ar.append("🤔 استراتيجية الطُعم: خسارة طفيفة مع حجم مبيعات ضخم. يرجى التأكد أن هذا مقصود كحملة تسويقية (Loss Leader).")
            rec_en.append("🤔 Loss Leader Check: High volume with slight loss. Verify if this is an intentional marketing strategy.")
        elif discount >= 0.15:
            rec_ar.append(f"✂️ إلغاء الخصم ({discount*100:.0f}%): الخصم يأكل رأس المال التشغيلي وليس الهامش فقط.")
            rec_en.append(f"✂️ Revoke {discount*100:.0f}% Discount: It's eroding operational capital.")
        elif sales > 500:
            rec_ar.append("🛑 إيقاف البيع مؤقتاً: اقتصاديات وحدة سلبية. المزيد من المبيعات يعني إفلاس أسرع.")
            rec_en.append("🛑 Halt Sales: Negative unit economics. Selling more means bleeding faster.")
        else:
            rec_ar.append("📉 تدقيق التكاليف (COGS): خسارة بدون خصم تعني أن تكلفة الشراء/الشحن تجاوزت سعر البيع.")
            rec_en.append("📉 COGS Audit: Loss without discount means sourcing/shipping costs exceed retail price.")

    # 4. المخزون الميت (Dead Inventory) وفخ المعاملات الصغيرة
    if profit > 0:
        if discount >= 0.40 and qty <= 2:
            rec_ar.append("🗑️ مخزون ميت: خصم ضخم (>40%) ولا يوجد سحب. توقف عن حرق العلامة التجارية وقم بتصفية المنتج.")
            rec_en.append("🗑️ Dead Inventory: Massive discount (>40%) but no volume. Liquidate immediately.")
        elif qty > 50 and sales < 50:
            rec_ar.append("📦 فخ التشغيل: كمية ضخمة بإيراد هزيل. تكلفة التغليف والشحن ستتجاوز قيمة الفاتورة.")
            rec_en.append("📦 Operational Trap: High qty, micro-revenue. Packaging/shipping will exceed item value.")

    # 5. الفرص الضائعة وتعظيم الأرباح
    if margin > 0.40 and qty < 5 and profit > 0:
        rec_ar.append("🚀 فرصة نمو: هامش ربح ممتاز (>40%) بكمية قليلة. ضاعف ميزانية التسويق لهذا المنتج فوراً.")
        rec_en.append("🚀 Growth Opportunity: Excellent margin (>40%) but low volume. Double marketing spend here.")
    elif qty > 15 and margin < 0.05 and profit > 0:
        rec_ar.append("⚠️ بيع جملة غير مجدي: إجهاد للتشغيل بهامش (<5%). يجب إعادة التفاوض مع المورد.")
        rec_en.append("⚠️ Inefficient Bulk Sale: High volume but <5% margin. Renegotiate with supplier.")

    # 6. شبكة الأمان (Fallback)
    if not rec_ar:
        if profit > 0 and margin >= 0.15:
            rec_ar.append("✅ أداء صحي ومستقر: الحفاظ على سياسة التسعير الحالية.")
            rec_en.append("✅ Healthy & Stable: Maintain current pricing strategy.")
        elif profit > 0:
            rec_ar.append("👁️ أداء ضعيف: الهامش منخفض، راقب عن كثب.")
            rec_en.append("👁️ Underperforming: Low margin, monitor closely.")

    return {
        "en": " | ".join(rec_en),
        "ar": " | ".join(rec_ar)
    }

def run_full_ai_analysis(df: pd.DataFrame) -> dict:
    if df.empty: return {}
    monthly_sales, forecast_df = run_arima_forecast(df)
    df_with_zscore, anomalies = detect_anomalies(df)
    df_with_tax = calculate_tax_risk(df)
    
    total_sales = df["Sales"].sum()
    profit_margin = df["Profit"].sum() / total_sales if total_sales > 0 else 0
    
    # استدعاء المحرك الاستراتيجي
    strategy = get_prescriptive_directive(monthly_sales, forecast_df, len(anomalies), profit_margin)
    
    return {
        "monthly_sales": monthly_sales,
        "forecast_df": forecast_df,
        "df_with_zscore": df_with_zscore,
        "anomalies": anomalies,
        "df_with_tax": df_with_tax,
        "strategy": strategy,
        "total_potential_recovery": abs(anomalies[anomalies["Profit"] < 0]["Profit"].sum())
    }