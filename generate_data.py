
import csv
import random
from pathlib import Path

# Commercial truck manufacturers and models
trucks = {
    "Peterbilt": ["579", "589", "567"],
    "Kenworth": ["T680", "W900", "T880"],
    "Freightliner": ["Cascadia", "M2 106"],
    "Volvo": ["VNL", "VNR"]
}

# Create a folder for generated datasets
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Generate 1,000 synthetic truck sales
with open(data_folder / "truck_sales.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "sale_id",
        "manufacturer",
        "model",
        "purchase_cost",
        "sale_price",
        "gross_profit"
    ])

    for sale_id in range(1, 1001):
        manufacturer = random.choice(list(trucks.keys()))
        model = random.choice(trucks[manufacturer])

        purchase_cost = random.randint(95000, 190000)
        sale_price = purchase_cost + random.randint(8000, 45000)
        gross_profit = sale_price - purchase_cost

        writer.writerow([
            sale_id,
            manufacturer,
            model,
            purchase_cost,
            sale_price,
            gross_profit
        ])

print("Successfully generated 1,000 synthetic truck sales!")
