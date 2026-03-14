"""
Topaz DSS V2 - Comprehensive Simulation Engine
Covers: Demand Forecasting (3 models), Production, HR, Finance, Marketing, Transport
"""
import numpy as np
from historical_data import HISTORICAL_DATA

# ── Manual Constants (Tabelas de Gestão) ──────────────────────────────────────

# Tabela 5: Production times per unit (minutes)
MACHINING_TIME = {"p1": 60, "p2": 75, "p3": 120}
MIN_ASSEMBLY_TIME = {"p1": 100, "p2": 150, "p3": 300}
RAW_MATERIAL_PER_UNIT = {"p1": 1, "p2": 2, "p3": 3}

# Tabela 7: Machine hours per quarter by shift count
SHIFT_HOURS = {1: 588, 2: 1092, 3: 1638}
WORKERS_PER_MACHINE = {1: 4, 2: 8, 3: 12}

# Tabela 10: Production costs
SUPERVISION_PER_SHIFT = 12500
OVERHEAD_PER_MACHINE = 3500
MACHINE_OP_COST_HR = 8
PLANNING_COST_PER_UNIT = 1
QC_DEPARTMENT = 8000

# Tabela 12: Transport
CONTAINER_RENT_DAY = 650
CONTAINER_CAPACITY = {"p1": 500, "p2": 250, "p3": 125}
ATLANTIC_CROSSING = 8000
DISTANCE_NAFTA = 250
DISTANCE_INTERNET = 150
MAX_KM_DAY = 400

# Tabela 13: Storage
FACTORY_STORAGE_LIMIT = 2000
STORAGE_ADMIN_QUARTERLY = 12500
OUTSIDE_STORAGE_PER_UNIT = 2.5
PRODUCT_STORAGE_EU = 3.5
PRODUCT_STORAGE_NAFTA = 4.0
EMERGENCY_SURCHARGE = 0.10

# Tabela 15: Personnel costs
RECRUIT_SPECIALIZED = 2000
FIRE_SPECIALIZED = 5000
TRAIN_SPECIALIZED = 8500
RECRUIT_UNSKILLED = 1000
FIRE_UNSKILLED = 2000

# Tabela 16: Worker hours
BASE_HOURS = 420
SAT_HOURS = {1: 84, 2: 42, 3: 42}
SUN_HOURS = {1: 84, 2: 84, 3: 84}
SHIFT_SUBSIDY = {1: 0, 2: 1/3, 3: 2/3}

# Tabela 17: Minimums
MIN_UNSKILLED_HOURS = 350
MIN_SPECIALIZED_WAGE = 4
MIN_MGMT_BUDGET = 30000
UNSKILLED_WAGE_PCT = 0.65

# Tabela 18: Machines
MACHINE_COST = 350000
MACHINE_DEPOSIT = 175000
MACHINE_INSTALL = 175000
DEPRECIATION_RATE = 0.025
MACHINE_SELL_FEE = 70000

# Tabela 20: Financial
TAX_RATE = 0.30
FIXED_ADMIN = 30000
VARIABLE_ADMIN_RATE = 0.003
CREDIT_CONTROL_PER_UNIT = 1
INTERNET_CREDIT_CARD_RATE = 1  # €1 per unit

# Tabela 3: Agents
AGENT_MIN_SUPPORT = 5000
AGENT_RECRUIT_COST = 7500
AGENT_CANCEL_COST = 5000

# Tabela 4: Internet
INTERNET_SALES_PCT = 0.03
ISP_JOIN_COST = 7500
ISP_PORT_COST = 1000
ISP_CANCEL_COST = 5000

# Tabela 8: Scrap values
SCRAP_VALUE = {"p1": 40, "p2": 80, "p3": 120}

# Tabela 9: Warranty
WARRANTY_RETAIL = {"p1": 60, "p2": 150, "p3": 250}

# ── Helper: Get historical series ────────────────────────────────────────────

def _get_series(key_path, quarters=None):
    """Extract a time series from historical data. key_path is a dot-separated path."""
    if quarters is None:
        quarters = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    values = []
    for q in quarters:
        d = HISTORICAL_DATA[q]
        for k in key_path.split("."):
            d = d[k]
        values.append(d)
    return values


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMAND FORECASTING (3 Models × 3 Products × 3 Markets)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_demand(new_prices):
    """Returns dict with 3 model predictions for each product×market."""
    products = ["p1", "p2", "p3"]
    markets = ["ue", "nafta", "internet"]
    results = {}

    for prod in products:
        results[prod] = {}
        for mkt in markets:
            hist_prices = _get_series(f"prices.{prod}.{mkt}")
            hist_orders = _get_series(f"orders.{prod}.{mkt}")
            new_p = new_prices[prod][mkt]

            # Model 1: Price Elasticity (point estimate Q4→Q5)
            if hist_prices[-2] != 0:
                pct_dp = (hist_prices[-1] - hist_prices[-2]) / hist_prices[-2]
                pct_dd = (hist_orders[-1] - hist_orders[-2]) / hist_orders[-2] if hist_orders[-2] != 0 else 0
                elasticity = pct_dd / pct_dp if pct_dp != 0 else -1.5
            else:
                elasticity = -1.5
            pct_price_change = (new_p - hist_prices[-1]) / hist_prices[-1] if hist_prices[-1] != 0 else 0
            elasticity_pred = int(hist_orders[-1] * (1 + pct_price_change * elasticity))

            # Model 2: Linear Trend (OLS on Q1-Q5)
            x = np.arange(len(hist_orders))
            slope, intercept = np.polyfit(x, hist_orders, 1) if len(hist_orders) > 1 else (0, hist_orders[-1])
            trend_pred = int(intercept + slope * len(hist_orders))
            # Adjust for price difference from trend
            avg_price_change = np.mean(np.diff(hist_prices)) if len(hist_prices) > 1 else 0
            expected_trend_price = hist_prices[-1] + avg_price_change
            if expected_trend_price != 0:
                price_adj = (new_p - expected_trend_price) / expected_trend_price * elasticity
                trend_pred = int(trend_pred * (1 + price_adj))

            # Model 3: Weighted Moving Average (more weight on recent quarters)
            weights = np.array([1, 2, 3, 4, 5][:len(hist_orders)])
            wma = np.average(hist_orders, weights=weights)
            # Growth rate
            if len(hist_orders) >= 2:
                growth_rates = [(hist_orders[i] - hist_orders[i-1]) / hist_orders[i-1]
                                for i in range(1, len(hist_orders)) if hist_orders[i-1] != 0]
                avg_growth = np.mean(growth_rates) if growth_rates else 0
            else:
                avg_growth = 0
            wma_pred = int(wma * (1 + avg_growth))
            # Price adjustment
            if hist_prices[-1] != 0:
                wma_pred = int(wma_pred * (1 + pct_price_change * elasticity * 0.5))

            results[prod][mkt] = {
                "elasticity": max(0, elasticity_pred),
                "linear_trend": max(0, trend_pred),
                "weighted_ma": max(0, wma_pred),
                "consensus": max(0, int(np.mean([elasticity_pred, trend_pred, wma_pred]))),
                "hist_orders": hist_orders,
                "hist_prices": hist_prices,
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRODUCTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_production_plan(delivery_targets, mp_purchase, machines, shifts,
                               spec_workers=None, unsk_workers=None, assembly_times=None):
    """
    Calculate the feasible production plan based on constraints.
    
    The user does NOT set production volume directly.  They set:
        - delivery_targets: {p1: {ue: N, nafta: N, internet: N}, ...} (what they want to ship)
        - mp_purchase: tons of raw material to buy this quarter
    
    Production is limited by the TIGHTEST constraint:
        1. Raw Material available (Q5 stock + purchase)
        2. Machine hours (machines × shift hours)
        3. Assembly hours (specialized workers × available hours)
        4. Non-specialized workers required to operate machines
    """
    q5 = HISTORICAL_DATA["Q5"]
    if assembly_times is None:
        assembly_times = q5["assembly"]["assembly_times"]
    if spec_workers is None:
        spec_workers = q5["hr"]["specialized"]["available_next"]
    if unsk_workers is None:
        unsk_workers = q5["hr"]["non_specialized"]["available_next"]

    flags = []
    details = {}

    # ── What the company WANTS to produce ──
    # Target = sum of delivery targets per product (total across all markets)
    target = {}
    for p in ["p1", "p2", "p3"]:
        target[p] = sum(delivery_targets[p][m] for m in ["ue", "nafta", "internet"])
    total_target = sum(target.values())
    details["target"] = target

    # ── Resource capacities ──

    # 1. Raw Material
    rm_stock = q5["raw_material"]["final_inventory"]
    rm_total = rm_stock + mp_purchase
    rm_needed = sum(target[p] * RAW_MATERIAL_PER_UNIT[p] for p in ["p1", "p2", "p3"])
    rm_ratio = rm_total / rm_needed if rm_needed > 0 else 999
    details["rm_stock"] = rm_stock
    details["rm_purchase"] = mp_purchase
    details["rm_total"] = rm_total
    details["rm_needed"] = rm_needed
    details["rm_surplus"] = rm_total - rm_needed

    if rm_ratio >= 1:
        flags.append(("success", f"✅ Raw Material: {rm_total:,.0f} available vs {rm_needed:,.0f} needed (surplus: {rm_total - rm_needed:,.0f})"))
    else:
        flags.append(("error", f"⚠️ Raw Material Shortage: {rm_total:,.0f} available vs {rm_needed:,.0f} needed (deficit: {rm_needed - rm_total:,.0f})"))

    # 2. Machine hours
    total_machine_hrs = machines * SHIFT_HOURS[shifts]
    req_machining_mins = sum(target[p] * MACHINING_TIME[p] for p in ["p1", "p2", "p3"])
    req_machining_hrs = req_machining_mins / 60.0
    machine_ratio = total_machine_hrs / req_machining_hrs if req_machining_hrs > 0 else 999
    machine_util = req_machining_hrs / total_machine_hrs * 100 if total_machine_hrs > 0 else 0
    details["machine_hrs_available"] = total_machine_hrs
    details["machine_hrs_needed"] = req_machining_hrs
    details["machine_utilization"] = machine_util

    if machine_ratio >= 1:
        flags.append(("success", f"✅ Machining: {req_machining_hrs:,.0f} / {total_machine_hrs:,.0f} hrs ({machine_util:.1f}%)"))
    else:
        flags.append(("error", f"⚠️ Machine Overload: {req_machining_hrs:,.0f} hrs needed vs {total_machine_hrs:,.0f} available"))

    # 3. Assembly hours
    max_spec_hrs = spec_workers * (BASE_HOURS + SAT_HOURS[shifts] + SUN_HOURS[shifts])
    req_assembly_mins = sum(target[p] * assembly_times[p] for p in ["p1", "p2", "p3"])
    req_assembly_hrs = req_assembly_mins / 60.0
    assembly_ratio = max_spec_hrs / req_assembly_hrs if req_assembly_hrs > 0 else 999
    assembly_util = req_assembly_hrs / max_spec_hrs * 100 if max_spec_hrs > 0 else 0
    details["assembly_hrs_available"] = max_spec_hrs
    details["assembly_hrs_needed"] = req_assembly_hrs
    details["assembly_utilization"] = assembly_util

    if assembly_ratio >= 1:
        flags.append(("success", f"✅ Assembly: {req_assembly_hrs:,.0f} / {max_spec_hrs:,.0f} hrs ({assembly_util:.1f}%)"))
    else:
        flags.append(("error", f"⚠️ Assembly Overload: {req_assembly_hrs:,.0f} hrs needed vs {max_spec_hrs:,.0f} available ({spec_workers} workers)"))

    # 4. Non-specialized workers
    req_unskilled = machines * WORKERS_PER_MACHINE[shifts]
    details["unskilled_needed"] = req_unskilled
    details["unskilled_available"] = unsk_workers

    if req_unskilled <= unsk_workers:
        flags.append(("success", f"✅ Unskilled Workers: {req_unskilled} / {unsk_workers}"))
    else:
        flags.append(("error", f"⚠️ Need {req_unskilled} unskilled workers, only {unsk_workers} available"))

    # ── Determine the binding constraint ──
    constraint_ratio = min(rm_ratio, machine_ratio, assembly_ratio)
    if constraint_ratio >= 1:
        binding = "None (all OK)"
        scale = 1.0
    else:
        scale = constraint_ratio
        if rm_ratio == constraint_ratio:
            binding = "Raw Material"
        elif machine_ratio == constraint_ratio:
            binding = "Machine Hours"
        else:
            binding = "Assembly Hours"

    details["binding_constraint"] = binding
    details["scale_factor"] = scale

    # ── Calculate actual production plan ──
    reject_rate = 0.033
    production = {}
    net_production = {}
    for p in ["p1", "p2", "p3"]:
        # Scale target down if constrained, and add ~3.3% extra to cover rejects
        if scale < 1:
            feasible = int(target[p] * scale)
        else:
            feasible = target[p]
        # Order a bit more to cover rejects (~3.3%)
        gross_production = int(feasible * (1 + reject_rate))
        rejects = int(gross_production * reject_rate)
        net = gross_production - rejects

        production[p] = gross_production
        net_production[p] = net
        details[f"{p}_target"] = target[p]
        details[f"{p}_gross_production"] = gross_production
        details[f"{p}_estimated_rejects"] = rejects
        details[f"{p}_net_production"] = net
        details[f"{p}_surplus_deficit"] = net - target[p]

    details["production"] = production
    details["net_production"] = net_production

    # ── Production costs ──
    actual_machining_hrs = sum(production[p] * MACHINING_TIME[p] for p in ["p1", "p2", "p3"]) / 60.0
    prod_cost = (shifts * SUPERVISION_PER_SHIFT +
                 machines * OVERHEAD_PER_MACHINE +
                 actual_machining_hrs * MACHINE_OP_COST_HR +
                 sum(production[p] for p in ["p1", "p2", "p3"]) * PLANNING_COST_PER_UNIT +
                 QC_DEPARTMENT)
    details["production_overhead_cost"] = prod_cost

    # ── Raw material cost ──
    rm_price_per_ton = q5["economic"]["raw_material_price_usd"]["6months"] / 1000 / q5["economic"]["exchange_rate_usd_eur"]
    details["rm_purchase_cost"] = mp_purchase * rm_price_per_ton

    return flags, details


# ══════════════════════════════════════════════════════════════════════════════
# 3. HR SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_hr(spec_recruit, spec_fire, spec_train, unsk_recruit, unsk_fire, wage_hr, shifts):
    """Simulate HR costs and workforce changes."""
    q5 = HISTORICAL_DATA["Q5"]
    spec_start = q5["hr"]["specialized"]["available_next"]
    unsk_start = q5["hr"]["non_specialized"]["available_next"]

    spec_end = spec_start + spec_recruit - spec_fire
    unsk_end = unsk_start + unsk_recruit - unsk_fire

    # Costs
    recruit_cost = (spec_recruit * RECRUIT_SPECIALIZED +
                    unsk_recruit * RECRUIT_UNSKILLED)
    fire_cost = (spec_fire * FIRE_SPECIALIZED +
                 unsk_fire * FIRE_UNSKILLED)
    train_cost = spec_train * TRAIN_SPECIALIZED

    # Wage costs
    unsk_wage = wage_hr * UNSKILLED_WAGE_PCT
    spec_base_hrs = BASE_HOURS
    spec_total_hrs = spec_base_hrs + SAT_HOURS.get(shifts, 0) + SUN_HOURS.get(shifts, 0)
    spec_wage_cost = (spec_end * spec_base_hrs * wage_hr +
                      spec_end * SAT_HOURS.get(shifts, 0) * wage_hr * 1.5 +
                      spec_end * SUN_HOURS.get(shifts, 0) * wage_hr * 2.0)

    unsk_total_hrs = max(MIN_UNSKILLED_HOURS, BASE_HOURS)
    unsk_wage_cost = (unsk_end * unsk_total_hrs * unsk_wage +
                      unsk_end * SAT_HOURS.get(shifts, 0) * unsk_wage * 1.5 +
                      unsk_end * SUN_HOURS.get(shifts, 0) * unsk_wage * 2.0)

    shift_subsidy_cost = unsk_wage_cost * SHIFT_SUBSIDY.get(shifts, 0)

    return {
        "spec_start": spec_start, "spec_end": spec_end,
        "unsk_start": unsk_start, "unsk_end": unsk_end,
        "recruit_cost": recruit_cost,
        "fire_cost": fire_cost,
        "train_cost": train_cost,
        "spec_wage_cost": spec_wage_cost,
        "unsk_wage_cost": unsk_wage_cost,
        "shift_subsidy": shift_subsidy_cost,
        "total_hr_cost": recruit_cost + fire_cost + train_cost + spec_wage_cost + unsk_wage_cost + shift_subsidy_cost,
        "max_assembly_hrs": spec_end * spec_total_hrs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. FINANCE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_finance(revenue, total_costs, machines, medium_term_loan_change=0):
    """Simulate financial position and constraints."""
    q5 = HISTORICAL_DATA["Q5"]
    bs = q5["balance_sheet"]

    # Admin costs
    admin_fixed = FIXED_ADMIN
    admin_variable = revenue * VARIABLE_ADMIN_RATE
    total_admin = admin_fixed + admin_variable

    # Depreciation
    machine_value = bs["machines"]
    depreciation = machine_value * DEPRECIATION_RATE

    # Operating profit
    gross_margin = revenue - total_costs
    ebit = gross_margin - total_admin - depreciation

    # Interest
    base_rate = q5["economic"]["interest_rate"]["ue"] / 100 / 4  # quarterly
    overdraft_rate = base_rate + 0.04 / 4
    ml_rate = 0.12 / 4  # medium-term fixed 12% annual

    existing_overdraft = bs["overdraft"]
    existing_ml = bs["medium_term_loans"]
    new_ml = existing_ml + medium_term_loan_change

    interest_overdraft = existing_overdraft * overdraft_rate
    interest_ml = new_ml * ml_rate

    # Pre-tax profit
    ebt = ebit - interest_overdraft - interest_ml

    # Tax (annual, simplified quarterly estimate)
    tax = max(0, ebt * TAX_RATE) if ebt > 0 else 0

    net_income = ebt - tax

    # Credit limit (Tabela 19)
    credit_base = 0.5 * (machine_value + bs["material_inventory"] + bs["product_inventory"])
    credit_base += 0.9 * bs["clients"]
    credit_base -= bs["taxes_payable"]
    credit_base -= bs["suppliers"]
    overdraft_limit = max(0, credit_base)

    return {
        "revenue": revenue,
        "total_costs": total_costs,
        "gross_margin": gross_margin,
        "admin_costs": total_admin,
        "depreciation": depreciation,
        "ebit": ebit,
        "interest_overdraft": interest_overdraft,
        "interest_ml": interest_ml,
        "ebt": ebt,
        "tax": tax,
        "net_income": net_income,
        "overdraft_limit": overdraft_limit,
        "current_overdraft": existing_overdraft,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. MARKETING SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_marketing(institutional_ad, direct_ads, agents_plan, internet_ports, website_dev):
    """Simulate marketing costs."""
    q5 = HISTORICAL_DATA["Q5"]
    q5_agents = q5["agents"]

    total_ad = institutional_ad * 1000  # input in thousands
    for prod in ["p1", "p2", "p3"]:
        for mkt in ["ue", "nafta", "internet"]:
            total_ad += direct_ads[prod][mkt] * 1000

    # Agent costs
    agent_costs = 0
    for mkt in ["ue", "nafta", "internet"]:
        current = q5_agents[mkt]["next"]
        target = agents_plan.get(mkt, current)
        if target > current:
            agent_costs += (target - current) * AGENT_RECRUIT_COST
        elif target < current:
            agent_costs += (current - target) * AGENT_CANCEL_COST
        agent_costs += target * AGENT_MIN_SUPPORT

    # Internet costs
    internet_cost = internet_ports * ISP_PORT_COST + website_dev * 1000

    return {
        "advertising_cost": total_ad,
        "agent_cost": agent_costs,
        "internet_cost": internet_cost,
        "total_marketing": total_ad + agent_costs + internet_cost,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRANSPORT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_transport(deliveries):
    """Estimate transport costs based on deliveries."""
    # Calculate containers needed
    containers = {}
    for mkt in ["ue", "nafta", "internet"]:
        units_p1_equiv = (deliveries.get("p1", {}).get(mkt, 0) +
                          deliveries.get("p2", {}).get(mkt, 0) * 2 +
                          deliveries.get("p3", {}).get(mkt, 0) * 4)
        containers[mkt] = max(1, int(np.ceil(units_p1_equiv / 500))) if units_p1_equiv > 0 else 0

    # Transport costs
    # EU: avg distance from historical ~1381km, ~4 days travel
    eu_days = max(1, int(np.ceil(1381 / MAX_KM_DAY)))
    eu_cost = containers["ue"] * CONTAINER_RENT_DAY * eu_days

    # NAFTA: distance to port + Atlantic crossing
    nafta_land_days = max(1, int(np.ceil(DISTANCE_NAFTA / MAX_KM_DAY)))
    nafta_cost = (containers["nafta"] * CONTAINER_RENT_DAY * nafta_land_days +
                  containers["nafta"] * ATLANTIC_CROSSING)

    # Internet: distance 150km
    internet_days = max(1, int(np.ceil(DISTANCE_INTERNET / MAX_KM_DAY)))
    internet_cost = containers["internet"] * CONTAINER_RENT_DAY * internet_days

    return {
        "containers": containers,
        "eu_cost": eu_cost,
        "nafta_cost": nafta_cost,
        "internet_cost": internet_cost,
        "total_transport": eu_cost + nafta_cost + internet_cost,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. FULL INCOME STATEMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_income_statement(prices, demand_forecast, prod_details, hr_result, marketing_result, transport_result, finance_inputs):
    """Build a complete predicted Q6 Income Statement.
    
    prod_details comes from calculate_production_plan() and contains: 
        production, net_production, rm_purchase_cost, production_overhead_cost
    """
    production = prod_details["production"]
    net_prod = prod_details["net_production"]

    # Revenue: min(net_production allocated per market, demand) × price
    revenue = 0
    sales_detail = {}

    for prod in ["p1", "p2", "p3"]:
        sales_detail[prod] = {}
        available = net_prod[prod]

        total_demand = sum(demand_forecast[prod][mkt]["consensus"] for mkt in ["ue", "nafta", "internet"])
        for mkt in ["ue", "nafta", "internet"]:
            mkt_demand = demand_forecast[prod][mkt]["consensus"]
            # Allocate available proportionally to demand across markets
            if total_demand > 0:
                mkt_alloc = int(available * mkt_demand / total_demand)
            else:
                mkt_alloc = 0
            sold = min(mkt_alloc, mkt_demand)
            sales_detail[prod][mkt] = sold
            revenue += sold * prices[prod][mkt]

    # COGS
    raw_material_cost = prod_details["rm_purchase_cost"]
    cogs = raw_material_cost + hr_result["spec_wage_cost"] + hr_result["unsk_wage_cost"] + prod_details["production_overhead_cost"]

    gross_margin = revenue - cogs

    # Total expenses
    total_expenses = (marketing_result["total_marketing"] +
                      transport_result["total_transport"] +
                      hr_result["recruit_cost"] + hr_result["fire_cost"] + hr_result["train_cost"] +
                      hr_result["shift_subsidy"])

    q5 = HISTORICAL_DATA["Q5"]
    finance = simulate_finance(revenue, cogs + total_expenses, q5["machines"]["available"], finance_inputs.get("ml_change", 0))

    return {
        "revenue": revenue,
        "cogs": cogs,
        "gross_margin": gross_margin,
        "total_expenses": total_expenses,
        "net_income": finance["net_income"],
        "finance": finance,
        "sales_detail": sales_detail,
        "raw_material_cost": raw_material_cost,
    }

