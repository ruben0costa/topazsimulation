import streamlit as st
import pandas as pd
import numpy as np
from historical_data import HISTORICAL_DATA
from engine import (
    forecast_demand, calculate_production_plan, simulate_hr,
    simulate_finance, simulate_marketing, simulate_transport,
    build_income_statement, RAW_MATERIAL_PER_UNIT, WORKERS_PER_MACHINE
)

st.set_page_config(page_title="Topaz Q6 DSS", layout="wide", page_icon="💎")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; border: 1px solid #0f3460; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1a2e; border-radius: 8px 8px 0 0; padding: 10px 20px; color: #e0e0e0; }
    .stTabs [aria-selected="true"] { background-color: #0f3460; color: #00d4ff; }
    div[data-testid="stExpander"] { border: 1px solid #0f3460; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Topaz Management Simulation — Q6 Decision Support System")
st.caption("Multi-model demand forecasting × Production × HR × Finance × Marketing × Transport")

q5 = HISTORICAL_DATA["Q5"]
PRODUCTS = ["p1", "p2", "p3"]
MARKETS = ["ue", "nafta", "internet"]
MARKET_LABELS = {"ue": "🇪🇺 EU", "nafta": "🌎 NAFTA", "internet": "🌐 Internet"}
PROD_LABELS = {"p1": "Product 1", "p2": "Product 2", "p3": "Product 3"}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: All Decision Inputs
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("🎮 Q6 Decision Inputs")

# ── Prices ──
st.sidebar.subheader("💰 Pricing (€)")
prices = {}
for prod in PRODUCTS:
    prices[prod] = {}
    cols = st.sidebar.columns(3)
    for i, mkt in enumerate(MARKETS):
        prices[prod][mkt] = cols[i].number_input(
            f"{PROD_LABELS[prod]} {MARKET_LABELS[mkt]}",
            min_value=1, value=q5["prices"][prod][mkt], key=f"price_{prod}_{mkt}"
        )

# ── Delivery Targets (per product × per market) ──
st.sidebar.subheader("📦 Delivery Targets (per market)")
st.sidebar.caption("Set how many units you want to deliver to each market. Production is auto-calculated.")
delivery_targets = {}
for prod in PRODUCTS:
    delivery_targets[prod] = {}
    st.sidebar.markdown(f"**{PROD_LABELS[prod]}**")
    cols = st.sidebar.columns(3)
    for i, mkt in enumerate(MARKETS):
        default = q5["orders"][prod][mkt]
        delivery_targets[prod][mkt] = cols[i].number_input(
            f"{MARKET_LABELS[mkt]}", min_value=0, value=default,
            key=f"del_{prod}_{mkt}"
        )

# ── Raw Material Purchase (MP) ──
st.sidebar.subheader("🪨 Raw Material (Matéria Prima)")
rm_q5_stock = q5["raw_material"]["final_inventory"]
# Calculate how much MP the target deliveries would need
total_mp_needed = sum(
    sum(delivery_targets[p][m] for m in MARKETS) * RAW_MATERIAL_PER_UNIT[p]
    for p in PRODUCTS
)
st.sidebar.caption(f"Q5 stock: **{rm_q5_stock:,}** tons | Target needs: **{total_mp_needed:,}** tons")
mp_purchase = st.sidebar.number_input(
    "MP to Purchase (tons)", min_value=0,
    value=max(0, total_mp_needed - rm_q5_stock + 500),  # suggest enough + buffer
    step=500, key="mp_purchase"
)

# ── Factory Resources ──
st.sidebar.subheader("⚙️ Factory Resources")
shifts = st.sidebar.selectbox("Shifts", [1, 2, 3], index=0)
machines = st.sidebar.number_input("Machines", min_value=1, value=q5["machines"]["next_quarter"])

# ── HR ──
st.sidebar.subheader("👷 Human Resources")
spec_recruit = st.sidebar.number_input("Recruit Specialized", min_value=0, value=0, key="sr")
spec_fire = st.sidebar.number_input("Fire Specialized", min_value=0, value=0, key="sf")
spec_train = st.sidebar.number_input("Train Specialized", min_value=0, value=3, key="st")
unsk_recruit = st.sidebar.number_input("Recruit Unskilled", min_value=0, value=6, key="ur")
unsk_fire = st.sidebar.number_input("Fire Unskilled", min_value=0, value=0, key="uf")
wage_hr = st.sidebar.number_input("Specialized Wage €/hr", min_value=4.0, value=12.5, step=0.5, key="wh")

# ── Marketing ──
st.sidebar.subheader("📢 Marketing")
institutional_ad = st.sidebar.number_input("Institutional Ads (€'000)", min_value=0, value=25, key="ia")
direct_ads = {}
for prod in PRODUCTS:
    direct_ads[prod] = {}
    for mkt in MARKETS:
        direct_ads[prod][mkt] = 25  # default

agents_plan = {
    "ue": st.sidebar.number_input("EU Agents", min_value=0, value=q5["agents"]["ue"]["next"], key="ag_ue"),
    "nafta": st.sidebar.number_input("NAFTA Distributors", min_value=0, value=q5["agents"]["nafta"]["next"], key="ag_na"),
    "internet": st.sidebar.number_input("Internet Distributor", min_value=0, value=q5["agents"]["internet"]["next"], key="ag_in"),
}
internet_ports = st.sidebar.number_input("Internet Ports", min_value=0, value=9, key="ip")
website_dev = st.sidebar.number_input("Website Dev (€'000)", min_value=0, value=30, key="wd")

# ── Finance ──
st.sidebar.subheader("🏦 Finance")
ml_change = st.sidebar.number_input("Medium-Term Loan Change (€'000)", value=0, key="ml")

# ══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════
demand_forecast = forecast_demand(prices)

# HR first (to get spec_end, unsk_end for production calculation)
hr_result = simulate_hr(spec_recruit, spec_fire, spec_train, unsk_recruit, unsk_fire, wage_hr, shifts)

# Production auto-calculated from delivery targets + MP + constraints
prod_flags, prod_details = calculate_production_plan(
    delivery_targets, mp_purchase, machines, shifts,
    spec_workers=hr_result["spec_end"],
    unsk_workers=hr_result["unsk_end"],
)

marketing_result = simulate_marketing(institutional_ad, direct_ads, agents_plan, internet_ports, website_dev)

# Build delivery estimate for transport (use net production allocated proportionally)
transport_deliveries = {}
for prod in PRODUCTS:
    transport_deliveries[prod] = {}
    for mkt in MARKETS:
        transport_deliveries[prod][mkt] = min(
            delivery_targets[prod][mkt],
            prod_details[f"{prod}_net_production"] * delivery_targets[prod][mkt] /
            max(1, sum(delivery_targets[prod][m] for m in MARKETS))
        )
transport_result = simulate_transport(transport_deliveries)

income = build_income_statement(
    prices, demand_forecast, prod_details, hr_result, marketing_result,
    transport_result, {"ml_change": ml_change * 1000}
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD (Tabs)
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Demand Forecast", "🏭 Production", "👷 HR", "💰 Finance", "📢 Marketing", "🚛 Transport", "📈 Historical Trends"
])

# ── Tab 1: Demand Forecast ────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Multi-Model Demand Forecast: Q6 Predictions by Market")

    for prod in PRODUCTS:
        st.markdown(f"### {PROD_LABELS[prod]}")
        rows = []
        for mkt in MARKETS:
            r = demand_forecast[prod][mkt]
            rows.append({
                "Market": MARKET_LABELS[mkt],
                "Price (€)": prices[prod][mkt],
                "Q5 Orders": r["hist_orders"][-1],
                "📈 Elasticity Model": r["elasticity"],
                "📉 Linear Trend": r["linear_trend"],
                "📊 Weighted MA": r["weighted_ma"],
                "✅ Consensus": r["consensus"],
                "Your Target": delivery_targets[prod][mkt],
                "Gap": delivery_targets[prod][mkt] - r["consensus"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        total_consensus = sum(demand_forecast[prod][m]["consensus"] for m in MARKETS)
        total_target = sum(delivery_targets[prod][m] for m in MARKETS)
        col1, col2 = st.columns(2)
        col1.metric(f"Consensus Demand", f"{total_consensus:,}")
        col2.metric(f"Your Delivery Target", f"{total_target:,}",
                    delta=f"{'Over' if total_target > total_consensus else 'Under'} by {abs(total_target - total_consensus):,}")

    st.divider()
    st.subheader("🔍 Model Comparison Chart")
    chart_data = []
    for prod in PRODUCTS:
        for mkt in MARKETS:
            r = demand_forecast[prod][mkt]
            chart_data.append({"Product-Market": f"{PROD_LABELS[prod]} {MARKET_LABELS[mkt]}",
                               "Elasticity": r["elasticity"], "Linear Trend": r["linear_trend"],
                               "Weighted MA": r["weighted_ma"], "Consensus": r["consensus"]})
    chart_df = pd.DataFrame(chart_data).set_index("Product-Market")
    st.bar_chart(chart_df)

# ── Tab 2: Production ────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏭 Production Plan (Auto-Calculated)")
    st.info("**You set Delivery Targets and buy Raw Material.** Production is automatically calculated based on the tightest resource constraint.")

    # Binding constraint
    binding = prod_details["binding_constraint"]
    scale = prod_details["scale_factor"]
    if scale >= 1:
        st.success(f"🎯 **All resources sufficient** — full delivery targets can be met!")
    else:
        st.error(f"⚠️ **Binding Constraint: {binding}** — production scaled to {scale:.1%} of target")

    # Constraint flags
    for flag_type, flag_msg in prod_flags:
        if flag_type == "error":
            st.error(flag_msg)
        else:
            st.success(flag_msg)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Machine Utilization", f"{prod_details['machine_utilization']:.1f}%")
    col2.metric("Assembly Utilization", f"{prod_details['assembly_utilization']:.1f}%")
    col3.metric("Production Cost", f"€ {prod_details['production_overhead_cost']:,.0f}")

    # Raw Material balance
    st.divider()
    st.markdown("### 🪨 Raw Material Balance")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Q5 Stock", f"{prod_details['rm_stock']:,} t")
    rm_cols[1].metric("Purchased", f"{prod_details['rm_purchase']:,} t")
    rm_cols[2].metric("Total Available", f"{prod_details['rm_total']:,} t")
    rm_cols[3].metric("Needed for Target", f"{prod_details['rm_needed']:,} t",
                      delta=f"{'Surplus' if prod_details['rm_surplus'] >= 0 else 'Deficit'}: {abs(prod_details['rm_surplus']):,} t")
    st.metric("MP Purchase Cost", f"€ {prod_details['rm_purchase_cost']:,.0f}")

    # Production output table
    st.divider()
    st.markdown("### 📋 Production Output Breakdown")
    prod_rows = []
    for prod in PRODUCTS:
        total_target = sum(delivery_targets[prod][m] for m in MARKETS)
        prod_rows.append({
            "Product": PROD_LABELS[prod],
            "Delivery Target": total_target,
            "Gross Production": prod_details[f"{prod}_gross_production"],
            "Est. Rejects (~3.3%)": prod_details[f"{prod}_estimated_rejects"],
            "Net Output": prod_details[f"{prod}_net_production"],
            "Surplus / Deficit": prod_details[f"{prod}_surplus_deficit"],
        })
    st.dataframe(pd.DataFrame(prod_rows), hide_index=True, use_container_width=True)

# ── Tab 3: HR ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("👷 Human Resources Simulator")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Specialized Workers", f"{hr_result['spec_start']} → {hr_result['spec_end']}")
    col2.metric("Unskilled Workers", f"{hr_result['unsk_start']} → {hr_result['unsk_end']}")
    col3.metric("Max Assembly Hours", f"{hr_result['max_assembly_hrs']:,}")
    col4.metric("Total HR Cost", f"€ {hr_result['total_hr_cost']:,.0f}")

    # Check unskilled sufficiency
    req_unskilled = machines * WORKERS_PER_MACHINE[shifts]
    if hr_result['unsk_end'] < req_unskilled:
        st.error(f"⚠️ You need **{req_unskilled}** unskilled workers for {machines} machines × {shifts} shift(s), but only have **{hr_result['unsk_end']}**. Recruit **{req_unskilled - hr_result['unsk_end']}** more!")
    else:
        st.success(f"✅ Unskilled workers OK: {hr_result['unsk_end']} available ≥ {req_unskilled} needed")

    st.divider()
    hr_breakdown = pd.DataFrame({
        "Cost Category": ["Recruitment", "Firing", "Training", "Specialized Wages", "Unskilled Wages", "Shift Subsidy"],
        "Amount (€)": [hr_result["recruit_cost"], hr_result["fire_cost"], hr_result["train_cost"],
                       hr_result["spec_wage_cost"], hr_result["unsk_wage_cost"], hr_result["shift_subsidy"]],
    })
    st.dataframe(hr_breakdown, hide_index=True, use_container_width=True)

    st.markdown("### Historical Workforce")
    hr_hist = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Specialized": [HISTORICAL_DATA[q]["hr"]["specialized"]["available_next"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "Non-Specialized": [HISTORICAL_DATA[q]["hr"]["non_specialized"]["available_next"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
    }).set_index("Quarter")
    st.line_chart(hr_hist)

# ── Tab 4: Finance ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("💰 Predicted Q6 Income Statement & Financial Position")

    fin = income["finance"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"€ {income['revenue']:,.0f}")
    col2.metric("COGS", f"€ {income['cogs']:,.0f}")
    col3.metric("Gross Margin", f"€ {income['gross_margin']:,.0f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expenses", f"€ {income['total_expenses']:,.0f}")
    col2.metric("Depreciation", f"€ {fin['depreciation']:,.0f}")
    col3.metric("Net Income", f"€ {income['net_income']:,.0f}",
                delta=f"{'Profit' if income['net_income'] > 0 else 'Loss'}")

    st.divider()
    st.markdown("### Income Statement Detail")
    is_rows = [
        ("Revenue", income["revenue"]),
        ("Cost of Goods Sold", -income["cogs"]),
        ("**Gross Margin**", income["gross_margin"]),
        ("Marketing & Sales", -income["total_expenses"]),
        ("Admin Costs", -fin["admin_costs"]),
        ("Depreciation", -fin["depreciation"]),
        ("**EBIT**", fin["ebit"]),
        ("Interest (Overdraft)", -fin["interest_overdraft"]),
        ("Interest (Med-Term Loan)", -fin["interest_ml"]),
        ("**EBT**", fin["ebt"]),
        ("Tax", -fin["tax"]),
        ("**Net Income**", income["net_income"]),
    ]
    is_df = pd.DataFrame(is_rows, columns=["Line Item", "Amount (€)"])
    st.dataframe(is_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Credit & Liquidity")
    col1, col2 = st.columns(2)
    col1.metric("Current Overdraft", f"€ {fin['current_overdraft']:,.0f}")
    col2.metric("Overdraft Limit (est.)", f"€ {fin['overdraft_limit']:,.0f}")

    st.markdown("### Historical Revenue & Net Income")
    fin_hist = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Revenue": [HISTORICAL_DATA[q]["financials"]["revenue"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "Net Income": [HISTORICAL_DATA[q]["financials"]["net_income"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
    }).set_index("Quarter")
    st.line_chart(fin_hist)

# ── Tab 5: Marketing ────────────────────────────────────────────────────────
with tab5:
    st.subheader("📢 Marketing Cost Simulator")

    col1, col2, col3 = st.columns(3)
    col1.metric("Advertising", f"€ {marketing_result['advertising_cost']:,.0f}")
    col2.metric("Agents & Distributors", f"€ {marketing_result['agent_cost']:,.0f}")
    col3.metric("Internet Costs", f"€ {marketing_result['internet_cost']:,.0f}")

    st.metric("Total Marketing Cost", f"€ {marketing_result['total_marketing']:,.0f}")

    st.divider()
    st.markdown("### Agents & Distributors")
    agent_rows = []
    for mkt in MARKETS:
        agent_rows.append({
            "Market": MARKET_LABELS[mkt],
            "Current": q5["agents"][mkt]["next"],
            "Planned": agents_plan[mkt],
            "Change": agents_plan[mkt] - q5["agents"][mkt]["next"],
        })
    st.dataframe(pd.DataFrame(agent_rows), hide_index=True, use_container_width=True)

# ── Tab 6: Transport ────────────────────────────────────────────────────────
with tab6:
    st.subheader("🚛 Transport & Logistics Simulator")

    col1, col2, col3 = st.columns(3)
    col1.metric("EU Transport", f"€ {transport_result['eu_cost']:,.0f}")
    col2.metric("NAFTA Transport", f"€ {transport_result['nafta_cost']:,.0f}")
    col3.metric("Internet Transport", f"€ {transport_result['internet_cost']:,.0f}")

    st.metric("Total Transport Cost", f"€ {transport_result['total_transport']:,.0f}")

    st.divider()
    st.markdown("### Container Requirements")
    cont_rows = []
    for mkt in MARKETS:
        cont_rows.append({
            "Market": MARKET_LABELS[mkt],
            "Containers": transport_result["containers"][mkt],
        })
    st.dataframe(pd.DataFrame(cont_rows), hide_index=True, use_container_width=True)

# ── Tab 7: Historical Trends ────────────────────────────────────────────────
with tab7:
    st.subheader("📈 Historical Q1-Q5 Data Trends")

    st.markdown("### Price Evolution")
    for prod in PRODUCTS:
        with st.expander(f"{PROD_LABELS[prod]} Prices"):
            price_df = pd.DataFrame({
                "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                "EU": [HISTORICAL_DATA[q]["prices"][prod]["ue"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
                "NAFTA": [HISTORICAL_DATA[q]["prices"][prod]["nafta"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
                "Internet": [HISTORICAL_DATA[q]["prices"][prod]["internet"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
            }).set_index("Quarter")
            st.line_chart(price_df)

    st.markdown("### Order Volume Evolution")
    for prod in PRODUCTS:
        with st.expander(f"{PROD_LABELS[prod]} Orders"):
            order_df = pd.DataFrame({
                "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                "EU": [HISTORICAL_DATA[q]["orders"][prod]["ue"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
                "NAFTA": [HISTORICAL_DATA[q]["orders"][prod]["nafta"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
                "Internet": [HISTORICAL_DATA[q]["orders"][prod]["internet"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
            }).set_index("Quarter")
            st.line_chart(order_df)

    st.markdown("### Production Volume")
    prod_hist = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "P1 Produced": [HISTORICAL_DATA[q]["production"]["p1"]["produced"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "P2 Produced": [HISTORICAL_DATA[q]["production"]["p2"]["produced"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "P3 Produced": [HISTORICAL_DATA[q]["production"]["p3"]["produced"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
    }).set_index("Quarter")
    st.line_chart(prod_hist)

    st.markdown("### Raw Material Usage")
    rm_hist = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Purchased": [HISTORICAL_DATA[q]["raw_material"]["purchased_current"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "Used": [HISTORICAL_DATA[q]["raw_material"]["used"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "End Stock": [HISTORICAL_DATA[q]["raw_material"]["final_inventory"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
    }).set_index("Quarter")
    st.line_chart(rm_hist)

    st.markdown("### Balance Sheet Summary")
    bs_hist = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "Total Assets": [HISTORICAL_DATA[q]["balance_sheet"]["total_assets"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "Total Equity": [HISTORICAL_DATA[q]["balance_sheet"]["total_equity"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
        "Overdraft": [HISTORICAL_DATA[q]["balance_sheet"]["overdraft"] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]],
    }).set_index("Quarter")
    st.line_chart(bs_hist)
