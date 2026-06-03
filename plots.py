"""
explore_prices.py
-----------------
Quick visualization of flat price distribution from flat_info.json
Run: python explore_prices.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("flat_info.json") as f:
    flats = json.load(f)

# Extract all prices
prices = []
for flat in flats:
    price_raw = flat.get("RENT/PER_MONTH_LETTINGS") or flat.get("PRICE")
    if not price_raw:
        continue
    try:
        price = float(price_raw.split("||||")[0])
        prices.append(price)
    except ValueError:
        continue

prices = np.array(prices)

print(f"Total flats with price: {len(prices)}")
print(f"Min  : €{prices.min():.2f}")
print(f"Max  : €{prices.max():.2f}")
print(f"Mean : €{prices.mean():.2f}")
print(f"Median: €{np.median(prices):.2f}")
print(f"\nPrice ranges:")
print(f"  < €100      : {(prices < 100).sum()} flats")
print(f"  €100-300    : {((prices >= 100) & (prices < 300)).sum()} flats")
print(f"  €300-500    : {((prices >= 300) & (prices < 500)).sum()} flats")
print(f"  €500-1000   : {((prices >= 500) & (prices < 1000)).sum()} flats")
print(f"  €1000-2000  : {((prices >= 1000) & (prices < 2000)).sum()} flats")
print(f"  €2000-5000  : {((prices >= 2000) & (prices < 5000)).sum()} flats")
print(f"  > €5000     : {(prices >= 5000).sum()} flats")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Vienna Flat Price Distribution", fontsize=14, fontweight='bold')

# Full range
axes[0].hist(prices, bins=100, color='steelblue', edgecolor='white', linewidth=0.3)
axes[0].set_title("Full Range (all prices)")
axes[0].set_xlabel("Price (€)")
axes[0].set_ylabel("Number of Flats")
axes[0].axvline(np.median(prices), color='red', linestyle='--', label=f'Median €{np.median(prices):.0f}')
axes[0].legend()

# Realistic range only
realistic = prices[(prices >= 300) & (prices <= 5000)]
axes[1].hist(realistic, bins=80, color='seagreen', edgecolor='white', linewidth=0.3)
axes[1].set_title(f"Realistic Range €300–€5000 ({len(realistic)} flats)")
axes[1].set_xlabel("Price (€)")
axes[1].set_ylabel("Number of Flats")
axes[1].axvline(np.median(realistic), color='red', linestyle='--', label=f'Median €{np.median(realistic):.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig("price_distribution.png", dpi=150)
plt.show()
print("\nSaved to price_distribution.png")