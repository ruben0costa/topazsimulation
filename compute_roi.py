import pandas as pd
import json

d1 = [4630, 2910, 1010]
p1 = [295, 425, 815]

d5 = [5560, 3320, 1220]
p5 = [270, 380, 730]

print("--- Analysis of Q1 vs Q5 ---")
for i in range(3):
    pct_change_demand = (d5[i] - d1[i]) / d1[i]
    pct_change_price = (p5[i] - p1[i]) / p1[i]
    elasticity = pct_change_demand / pct_change_price if pct_change_price != 0 else 0
    print(f"Product {i+1} : Demand changed by {pct_change_demand*100:.1f}%, Price changed by {pct_change_price*100:.1f}%. Price Elasticity: {elasticity:.2f}")

print("\n--- Marketing ROI Estimate ---")
print("Marketing spend remained mostly constant per region/product baseline (25k), but cumulative branding impacts demand. ROI modeled dynamically in Q6 predictor.")
