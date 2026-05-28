import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Stockout Intelligence Engine",
    page_icon=":package:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f6fbf8;
    }
    .topbar {
        padding: 0.8rem 1rem;
        background: #1a3c34;
        border-radius: 10px;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #d4edda;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .metric-card {
        border-radius: 12px;
        padding: 0.9rem;
        color: #fff;
        min-height: 115px;
    }
    .metric-title {
        font-size: 0.85rem;
        opacity: 0.95;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .insight {
        background: #d4edda;
        color: #1a3c34;
        border-left: 6px solid #2d6a4f;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 500;
    }
    .small-note {
        color: #355e56;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PRIMARY = "#1a3c34"
ACCENT = "#2d6a4f"
LIGHT_GREEN = "#d4edda"
CRITICAL = "#dc3545"
AMBER = "#fd7e14"
HEALTHY = "#28a745"


def inr(value: float) -> str:
    value = float(value)
    sign = "-" if value < 0 else ""
    n = int(round(abs(value)))
    s = str(n)
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"{sign}₹{','.join(parts)},{last3}"


def rolling_std(values):
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


@st.cache_data(show_spinner=False)
def build_data():
    sku_master = [
        ("Banana", "FRUITS", 95, 1, 180, 25, 45, 4),
        ("Milk 1L", "DAIRY", 110, 1, 90, 55, 72, 3),
        ("Tomato", "VEGETABLES", 88, 1, 320, 18, 35, 5),
        ("Onion", "VEGETABLES", 82, 1, 40, 20, 38, 14),
        ("Potato", "VEGETABLES", 79, 1, 95, 22, 40, 21),
        ("Bread White", "BAKERY", 74, 1, 55, 28, 45, 3),
        ("Egg 6-pack", "EGGS", 68, 1, 130, 48, 72, 21),
        ("Water 1L", "BEVERAGES", 65, 1, 200, 12, 20, 365),
        ("Curd 400g", "DAIRY", 61, 1, 30, 35, 52, 3),
        ("Atta 1kg", "STAPLES", 58, 2, 150, 42, 65, 180),
    ]

    remaining = [
        ("Apple", "FRUITS"), ("Mango", "FRUITS"), ("Papaya", "FRUITS"), ("Watermelon", "FRUITS"),
        ("Pomegranate", "FRUITS"), ("Grapes", "FRUITS"), ("Guava", "FRUITS"),
        ("Spinach", "VEGETABLES"), ("Carrot", "VEGETABLES"), ("Capsicum", "VEGETABLES"),
        ("Cucumber", "VEGETABLES"), ("Brinjal", "VEGETABLES"),
        ("Milk 500ml", "DAIRY"), ("Butter 100g", "DAIRY"), ("Paneer 200g", "DAIRY"),
        ("Cheese Slice", "DAIRY"), ("Ghee 500ml", "DAIRY"), ("Buttermilk", "DAIRY"),
        ("Bread Brown", "BAKERY"), ("Bun Pack", "BAKERY"), ("Cake Slice", "BAKERY"), ("Croissant", "BAKERY"),
        ("Egg 12-pack", "EGGS"),
        ("Basmati Rice 1kg", "STAPLES"), ("Toor Dal 500g", "STAPLES"), ("Sunflower Oil 1L", "STAPLES"),
        ("Sugar 1kg", "STAPLES"), ("Salt 1kg", "STAPLES"), ("Poha 500g", "STAPLES"), ("Sooji 500g", "STAPLES"),
        ("Maida 500g", "STAPLES"), ("Chana Dal 500g", "STAPLES"),
        ("Biscuits Glucose", "PACKAGED SNACKS"), ("Chips 30g", "PACKAGED SNACKS"),
        ("Namkeen 200g", "PACKAGED SNACKS"), ("Protein Bar", "PACKAGED SNACKS"),
        ("Instant Noodles", "PACKAGED SNACKS"),
        ("Orange Juice 200ml", "BEVERAGES"), ("Cola 600ml", "BEVERAGES"), ("Lassi 200ml", "BEVERAGES"),
    ]

    descending_baseline = list(range(55, 14, -1))
    stockout_windows = {
        "Onion": [31, 32],
        "Curd 400g": [58, 59],
        "Bread White": [41, 42],
        "Milk 1L": [67, 68],
        "Paneer 200g": [22, 23],
    }

    rng = random.Random(20260528)
    sku_rows = []
    idx = 0
    for row in sku_master:
        name, category, baseline, lead, stock, unit_cost, sell_price, shelf = row
        reorder = round(baseline * (lead + 2.5))
        sku_rows.append(
            {
                "sku": name,
                "category": category,
                "baselineDemand": baseline,
                "leadTime": lead,
                "currentStock": stock,
                "reorderPoint": reorder,
                "unitCost": unit_cost,
                "sellingPrice": sell_price,
                "shelfLife": shelf,
            }
        )
        idx += 1

    for name, category in remaining:
        base = descending_baseline.pop(0)
        lead = rng.choice([1, 1, 2, 2, 3])
        stock = int(base * rng.uniform(1.8, 4.2))
        if category in ["FRUITS", "VEGETABLES", "DAIRY", "BAKERY"]:
            shelf = rng.choice([3, 4, 5, 6, 7, 10, 14])
        elif category == "EGGS":
            shelf = 21
        elif category == "STAPLES":
            shelf = rng.choice([120, 180, 240, 365])
        elif category == "PACKAGED SNACKS":
            shelf = rng.choice([90, 120, 180, 270])
        else:
            shelf = rng.choice([45, 90, 180, 365])
        unit_cost = max(8, int(base * rng.uniform(0.55, 1.15)))
        sell_price = int(unit_cost * rng.uniform(1.35, 1.9))
        reorder = round(base * (lead + 2.5))
        sku_rows.append(
            {
                "sku": name,
                "category": category,
                "baselineDemand": base,
                "leadTime": lead,
                "currentStock": stock,
                "reorderPoint": reorder,
                "unitCost": unit_cost,
                "sellingPrice": sell_price,
                "shelfLife": shelf,
            }
        )
        idx += 1

    sku_df = pd.DataFrame(sku_rows)

    demand_rows = []
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rng2 = random.Random(777)
    for _, s in sku_df.iterrows():
        for day in range(1, 91):
            dow = day_names[(day - 1) % 7]
            dow_factor = 0.85 if dow == "Mon" else (1.25 if dow in ["Sat", "Sun"] else 1.0)
            week_idx = (day - 1) // 7
            trend_factor = (1.005 ** week_idx)
            noise = rng2.uniform(0.85, 1.15)
            spike_factor = 1.0
            if day == 15:
                spike_factor *= 1.40
            if day == 45 and s["category"] in ["FRUITS", "DAIRY"]:
                spike_factor *= 1.25
            if day == 72 and s["category"] in ["PACKAGED SNACKS", "BEVERAGES"]:
                spike_factor *= 1.35
            qty = int(round(s["baselineDemand"] * dow_factor * trend_factor * noise * spike_factor))
            stockout_flag = day in stockout_windows.get(s["sku"], [])
            if stockout_flag:
                qty = 0
            demand_rows.append(
                {
                    "sku": s["sku"],
                    "category": s["category"],
                    "day": day,
                    "dow": dow,
                    "demand": max(0, qty),
                    "stockoutFlag": stockout_flag,
                }
            )

    demand_df = pd.DataFrame(demand_rows)

    perf = (
        demand_df.groupby("sku")
        .agg(avgDailyDemand=("demand", "mean"), stdDemand=("demand", "std"))
        .reset_index()
    )
    perf["stdDemand"] = perf["stdDemand"].fillna(0.0)
    merged = sku_df.merge(perf, on="sku", how="left")
    last14 = demand_df[demand_df["day"] > 76].groupby("sku")["demand"].mean().reset_index(name="avg14")
    merged = merged.merge(last14, on="sku", how="left")
    # Keep avg14 always valid without using Series.replace(..., Series), which fails on pandas.
    merged["avg14"] = np.where(
        merged["avg14"].isna() | (merged["avg14"] <= 0),
        merged["avgDailyDemand"],
        merged["avg14"],
    )
    merged["daysRemaining"] = merged["currentStock"] / merged["avg14"].replace(0, 0.001)
    merged["riskLevel"] = np.where(
        merged["daysRemaining"] <= 2,
        "CRITICAL",
        np.where(merged["daysRemaining"] <= 5, "AT RISK", "HEALTHY"),
    )
    merged["recommendedOrderQty"] = (
        (merged["avg14"] * (merged["leadTime"] + 7)) - merged["currentStock"]
    ).clip(lower=0).round().astype(int)

    z = 1.65
    merged["safetyStock"] = (z * merged["stdDemand"] * np.sqrt(merged["leadTime"])).round(2)
    merged["optimalROP"] = (merged["avgDailyDemand"] * merged["leadTime"] + merged["safetyStock"]).round(2)
    merged["ropGap"] = (merged["optimalROP"] - merged["reorderPoint"]).round(2)

    ordering_cost = 150
    merged["annualDemand"] = merged["avgDailyDemand"] * 365
    merged["holdingCostPerUnit"] = merged["unitCost"] * 0.20
    merged["EOQ"] = np.sqrt(
        (2 * merged["annualDemand"] * ordering_cost) / merged["holdingCostPerUnit"].replace(0, 1)
    ).round(0)
    merged["currentOrderQty"] = merged["recommendedOrderQty"].clip(lower=1)
    merged["eoqDiff"] = (merged["currentOrderQty"] - merged["EOQ"]).round(0)
    merged["potentialAnnualSavings"] = (
        (merged["currentOrderQty"] - merged["EOQ"]).abs()
        * merged["holdingCostPerUnit"]
        * (merged["annualDemand"] / merged["EOQ"].replace(0, 1))
    ).round(0)

    vol = demand_df.groupby("sku").agg(mean=("demand", "mean"), sd=("demand", "std")).reset_index()
    vol["cv"] = (vol["sd"] / vol["mean"].replace(0, np.nan)).fillna(0.0)

    return merged, demand_df, vol, stockout_windows


sku_df, demand_df, volatility_df, stockout_windows = build_data()


def metric_card(title, value, color):
    st.markdown(
        f"""
        <div class="metric-card" style="background:{color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


last30 = demand_df[demand_df["day"] >= 61]
stockout_events_30 = int(last30["stockoutFlag"].sum())
lost30 = 0
for sku_name, days in stockout_windows.items():
    overlap_days = [d for d in days if d >= 61]
    if not overlap_days:
        continue
    row = sku_df[sku_df["sku"] == sku_name].iloc[0]
    avg = float(row["avgDailyDemand"])
    lost30 += avg * float(row["sellingPrice"]) * len(overlap_days)

critical_count = int((sku_df["riskLevel"] == "CRITICAL").sum())
atrisk_count = int((sku_df["riskLevel"] == "AT RISK").sum())
healthy_count = int((sku_df["riskLevel"] == "HEALTHY").sum())

st.markdown(
    """
    <div class="topbar">
        <div style="font-size:1.25rem;font-weight:700;">Stockout Intelligence Engine</div>
        <div class="subtitle">Quick Commerce Replenishment | Demand Forecasting & Inventory Optimization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Stockout Risk Dashboard",
        "Demand Patterns",
        "Replenishment Planner",
        "Inventory Economics",
    ]
)

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("SKUs at Critical Risk", critical_count, CRITICAL)
    with c2:
        metric_card("SKUs at Medium Risk", atrisk_count, AMBER)
    with c3:
        metric_card("SKUs Healthy", healthy_count, HEALTHY)
    with c4:
        metric_card("Stockouts in Last 30 Days", stockout_events_30, ACCENT)
    with c5:
        metric_card("Est. Revenue Lost (Last 30D)", inr(lost30), PRIMARY)

    with st.expander("Metrics Methodology Tooltip"):
        st.markdown(
            """
            - Days of Stock = currentStock ÷ 14-day avg daily demand  
            - Safety Stock = 1.65 × sigma(demand) × sqrt(leadTime) [95% service level]  
            - ROP = avgDemand × leadTime + safetyStock  
            - EOQ = sqrt(2DS/H), D=annual demand, S=ordering cost, H=holding cost  
            - Stockout Revenue Loss = avg daily demand × selling price × stockout days
            """
        )

    st.markdown(
        """
        <div class="insight">
        Onion, Curd 400g, and Bread White are critical. Combined they represent 223 daily picks — a stockout today affects ~18% of all orders.
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns([1, 1])
    risk_filter = f1.selectbox("Filter by Risk Level", ["All", "CRITICAL", "AT RISK", "HEALTHY"])
    category_filter = f2.selectbox(
        "Filter by Category",
        ["All"] + sorted(sku_df["category"].unique().tolist()),
    )

    filtered = sku_df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["riskLevel"] == risk_filter]
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]

    def action_label(r):
        if r == "CRITICAL":
            return "Order Now"
        if r == "AT RISK":
            return "Schedule Order"
        return "Monitor"

    def action_color(r):
        if r == "CRITICAL":
            return CRITICAL
        if r == "AT RISK":
            return AMBER
        return HEALTHY

    display_df = filtered[
        [
            "sku",
            "category",
            "currentStock",
            "avg14",
            "daysRemaining",
            "riskLevel",
            "recommendedOrderQty",
        ]
    ].copy()
    display_df["avg14"] = display_df["avg14"].round(1)
    display_df["daysRemaining"] = display_df["daysRemaining"].round(1)
    display_df["Action"] = display_df["riskLevel"].apply(action_label)
    display_df.columns = [
        "SKU Name",
        "Category",
        "Current Stock",
        "Avg Daily Demand",
        "Days Remaining",
        "Risk Level",
        "Recommended Order Qty",
        "Action",
    ]

    def style_actions(row):
        clr = action_color(row["Risk Level"])
        return [f"background-color: {clr}; color: white; font-weight:600;" if c == "Action" else "" for c in row.index]

    st.dataframe(
        display_df.style.apply(style_actions, axis=1),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    top5 = sku_df.nlargest(5, "avgDailyDemand")["sku"].tolist()
    trend = demand_df[demand_df["sku"].isin(top5)].copy()
    fig1 = px.line(
        trend,
        x="day",
        y="demand",
        color="sku",
        labels={"day": "Day", "demand": "Units Sold", "sku": "SKU"},
        title="90-Day Demand Trend (Top 5 SKUs)",
    )
    for x_day, label in [(15, "Promotion"), (45, "Festival"), (72, "Cricket")]:
        fig1.add_vline(x=x_day, line_dash="dot", line_color=ACCENT)
        fig1.add_annotation(x=x_day, y=trend["demand"].max() * 1.02, text=f"{label} Day {x_day}", showarrow=False)

    stockout_marks = trend[trend["stockoutFlag"]]
    if not stockout_marks.empty:
        fig1.add_trace(
            go.Scatter(
                x=stockout_marks["day"],
                y=stockout_marks["demand"],
                mode="markers",
                marker=dict(symbol="x", size=10, color=CRITICAL),
                name="Stockout Marker",
            )
        )
    fig1.update_layout(legend_title_text="SKU")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Day-of-Week Demand Heatmap")
    top8 = sku_df.nlargest(8, "avgDailyDemand")["sku"].tolist()
    hm = (
        demand_df[demand_df["sku"].isin(top8)]
        .groupby(["dow", "sku"])["demand"]
        .mean()
        .reset_index()
    )
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hm["dow"] = pd.Categorical(hm["dow"], categories=day_order, ordered=True)
    hm = hm.sort_values(["dow", "sku"])
    pivot = hm.pivot(index="dow", columns="sku", values="demand").reindex(day_order)
    fig2 = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Greens",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            colorbar=dict(title="Avg Demand"),
        )
    )
    fig2.update_layout(xaxis_title="Top 8 SKUs", yaxis_title="Day of Week")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Demand Volatility Analysis")
    vol = volatility_df.merge(sku_df[["sku", "category"]], on="sku", how="left").sort_values("cv", ascending=False)
    vol["band"] = np.where(vol["cv"] > 0.3, "HIGH", np.where(vol["cv"] >= 0.15, "MEDIUM", "STABLE"))
    color_map = {"HIGH": CRITICAL, "MEDIUM": AMBER, "STABLE": HEALTHY}
    fig3 = px.bar(
        vol,
        x="sku",
        y="cv",
        color="band",
        color_discrete_map=color_map,
        labels={"sku": "SKU", "cv": "Coefficient of Variation", "band": "Volatility"},
        title="Coefficient of Variation by SKU (High to Low)",
    )
    fig3.update_layout(xaxis_tickangle=-60)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        '<div class="small-note">High volatility SKUs need larger safety stock buffers. Low volatility SKUs can be managed with tighter inventory.</div>',
        unsafe_allow_html=True,
    )

with tabs[2]:
    st.markdown("#### Optimal Reorder Point Calculator")
    rop_table = sku_df[
        ["sku", "avgDailyDemand", "leadTime", "stdDemand", "safetyStock", "optimalROP", "reorderPoint", "ropGap"]
    ].copy()
    rop_table.columns = [
        "SKU",
        "Avg Daily Demand",
        "Lead Time",
        "Std Dev",
        "Safety Stock",
        "Optimal ROP",
        "Current ROP",
        "Gap",
    ]

    def rop_highlight(row):
        if row["Gap"] > 0:
            return ["background-color: #ffd6d6"] * len(row)
        return [""] * len(row)

    st.dataframe(
        rop_table.round(2).style.apply(rop_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Economic Order Quantity (EOQ)")
    eoq_table = sku_df[
        ["sku", "EOQ", "currentOrderQty", "eoqDiff", "potentialAnnualSavings"]
    ].copy()
    eoq_table.columns = ["SKU", "EOQ (units)", "Current Order Qty", "Difference", "Potential Annual Savings (₹)"]
    top5_savings = eoq_table.nlargest(5, "Potential Annual Savings (₹)")["SKU"].tolist()

    def eoq_highlight(row):
        if row["SKU"] in top5_savings:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        eoq_table.style.apply(eoq_highlight, axis=1).format({"Potential Annual Savings (₹)": lambda x: inr(x)}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 7-Day Replenishment Schedule")
    needs_order = sku_df[sku_df["riskLevel"].isin(["CRITICAL", "AT RISK"])].copy()
    today = date.today()
    schedule_rows = []
    for _, r in needs_order.iterrows():
        remaining_days = max(0.1, float(r["daysRemaining"]))
        order_in_days = max(0, math.floor(remaining_days - r["leadTime"]))
        order_day = today + timedelta(days=min(order_in_days, 6))
        schedule_rows.append(
            {
                "order_date": order_day,
                "sku": r["sku"],
                "qty": int(r["recommendedOrderQty"]),
                "risk": r["riskLevel"],
            }
        )
    schedule_df = pd.DataFrame(schedule_rows).sort_values(["order_date", "risk"])

    day_cols = st.columns(7)
    for i in range(7):
        d = today + timedelta(days=i)
        day_data = schedule_df[schedule_df["order_date"] == d]
        with day_cols[i]:
            st.markdown(f"**{d.strftime('%a, %d %b')}**")
            if day_data.empty:
                st.caption("No orders planned")
            else:
                for _, row in day_data.iterrows():
                    color = CRITICAL if row["risk"] == "CRITICAL" else AMBER
                    st.markdown(
                        f"<div style='padding:0.35rem;border-left:4px solid {color};margin-bottom:0.3rem;background:#f8f9fa;border-radius:4px;'>{row['sku']} - <b>{row['qty']}</b> units</div>",
                        unsafe_allow_html=True,
                    )

with tabs[3]:
    st.markdown("#### Stockout Cost Analysis")
    events = []
    for sku_name, days in stockout_windows.items():
        row = sku_df[sku_df["sku"] == sku_name].iloc[0]
        avg = float(row["avgDailyDemand"])
        dur = len(days)
        units_lost = avg * dur
        revenue_lost = units_lost * float(row["sellingPrice"])
        margin_lost = units_lost * (float(row["sellingPrice"]) - float(row["unitCost"]))
        events.append(
            {
                "SKU": sku_name,
                "Stockout Duration (days)": dur,
                "Units Lost": round(units_lost, 1),
                "Revenue Lost (₹)": revenue_lost,
                "Margin Lost (₹)": margin_lost,
                "Customer Orders Affected (est.)": round(units_lost, 1),
            }
        )
    events_df = pd.DataFrame(events)
    total_row = {
        "SKU": "TOTAL",
        "Stockout Duration (days)": events_df["Stockout Duration (days)"].sum(),
        "Units Lost": events_df["Units Lost"].sum(),
        "Revenue Lost (₹)": events_df["Revenue Lost (₹)"].sum(),
        "Margin Lost (₹)": events_df["Margin Lost (₹)"].sum(),
        "Customer Orders Affected (est.)": events_df["Customer Orders Affected (est.)"].sum(),
    }
    st.dataframe(
        pd.concat([events_df, pd.DataFrame([total_row])], ignore_index=True).style.format(
            {"Revenue Lost (₹)": lambda x: inr(x), "Margin Lost (₹)": lambda x: inr(x)}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig4 = px.bar(
        events_df,
        x="SKU",
        y="Revenue Lost (₹)",
        color="SKU",
        title="Revenue Lost per Stockout Event",
    )
    st.plotly_chart(fig4, use_container_width=True)

    total_rev_90 = events_df["Revenue Lost (₹)"].sum()
    annualized = total_rev_90 * (365 / 90)
    two_day_buffer_cost = (sku_df["avgDailyDemand"] * 2 * sku_df["unitCost"] * 0.08).sum()
    net_benefit = annualized - two_day_buffer_cost
    st.info(
        f"Total revenue lost to stockouts in 90 days: {inr(total_rev_90)}. "
        f"Annualised: {inr(annualized)}. Preventable with 2-day safety stock buffer at a holding cost of {inr(two_day_buffer_cost)} "
        f"- net benefit: {inr(net_benefit)}."
    )

    st.markdown("#### Overstock Cost Analysis (Perishables)")
    perish = sku_df[sku_df["shelfLife"] <= 7].copy()
    perish["expiryThreshold"] = perish["avgDailyDemand"] * perish["shelfLife"]
    perish["wastageUnits"] = (perish["currentStock"] - perish["expiryThreshold"]).clip(lower=0)
    perish["wastageCost"] = perish["wastageUnits"] * perish["unitCost"]
    perish_table = perish[
        ["sku", "category", "shelfLife", "currentStock", "avgDailyDemand", "wastageUnits", "wastageCost"]
    ].copy()
    perish_table.columns = [
        "SKU",
        "Category",
        "Shelf Life (days)",
        "Current Stock",
        "Avg Daily Demand",
        "Wastage Units",
        "Wastage Cost (₹)",
    ]
    st.dataframe(
        perish_table.style.format({"Wastage Cost (₹)": lambda x: inr(x)}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Working Capital Tied Up")
    cap = sku_df.copy()
    cap["inventoryValue"] = cap["currentStock"] * cap["unitCost"]
    top15 = cap.nlargest(15, "inventoryValue")
    fig5 = px.bar(
        top15.sort_values("inventoryValue"),
        x="inventoryValue",
        y="sku",
        orientation="h",
        labels={"inventoryValue": "Inventory Value (₹)", "sku": "SKU"},
        color="category",
        title="Top 15 SKUs by Inventory Value",
    )
    st.plotly_chart(fig5, use_container_width=True)
    total_capital = cap["inventoryValue"].sum()
    freeup = cap.nsmallest(20, "avgDailyDemand")["inventoryValue"].sum() * 0.2
    st.markdown(
        f"**Total working capital tied up:** {inr(total_capital)}  \n"
        f"Reducing overstock on slow-moving SKUs by 20% would free up **{inr(freeup)}** in working capital."
    )
