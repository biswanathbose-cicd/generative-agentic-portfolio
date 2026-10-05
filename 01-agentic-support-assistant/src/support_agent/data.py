"""Deterministic synthetic catalog + orders.

All data is generated (seeded) - there is no real retailer data in this repository.
Run ``python -m support_agent.data`` to regenerate ``data/catalog.json`` and ``data/orders.json``.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Fixed "today" so every run (tests, evals, docs) is reproducible.
REFERENCE_TODAY = date(2026, 10, 4)
RETURN_WINDOW_DAYS = 30
NON_RETURNABLE_CATEGORIES = {"grocery"}

NOUNS = {
    "electronics": ["wireless earbuds", "bluetooth speaker", "laptop stand", "usb-c charger",
                    "smart watch", "gaming mouse", "hdmi cable", "tablet case"],
    "home": ["air fryer", "blender", "vacuum cleaner", "memory foam pillow",
             "led desk lamp", "coffee maker", "bath towel set", "storage bin"],
    "grocery": ["oats cereal", "almond milk", "whole bean coffee", "olive oil",
                "trail mix", "sparkling water", "peanut butter", "pasta sauce"],
    "toys": ["building blocks set", "remote control car", "plush bear", "jigsaw puzzle",
             "board game", "art supplies kit", "water gun", "kids scooter"],
    "apparel": ["running shoes", "rain jacket", "cotton t-shirt", "denim jeans",
                "winter gloves", "wool socks", "baseball cap", "yoga pants"],
}
ADJECTIVES = ["compact", "premium", "budget", "family-size", "lightweight", "durable",
              "waterproof", "rechargeable", "organic", "classic", "ultra", "everyday"]
BRANDS = ["Brightwell", "Northpeak", "Homeora", "Kidzo", "Fresh Harvest", "Voltix", "Oakline"]
FEATURES = {
    "electronics": ["long battery life", "fast charging", "noise isolation", "compact design",
                    "works with most devices", "one year warranty"],
    "home": ["easy to clean", "energy efficient", "quiet operation", "space saving",
             "dishwasher safe parts", "two year warranty"],
    "grocery": ["no artificial flavors", "resealable pack", "great for breakfast",
                "gluten free option", "non-gmo", "pantry staple"],
    "toys": ["ages 6 and up", "encourages creativity", "easy assembly", "safe materials",
             "great gift idea", "batteries included"],
    "apparel": ["machine washable", "breathable fabric", "true to size", "all season wear",
                "reinforced stitching", "available in many colors"],
}
PRICE_RANGE = {"electronics": (12, 180), "home": (15, 150), "grocery": (2, 15),
               "toys": (8, 60), "apparel": (10, 90)}


def build_catalog(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    products: list[dict] = []
    n = 0
    for category, nouns in NOUNS.items():
        for noun in nouns:
            combos = rng.sample([(a, b) for a in ADJECTIVES for b in BRANDS], 3)
            for adj, brand in combos:
                n += 1
                lo, hi = PRICE_RANGE[category]
                feats = rng.sample(FEATURES[category], 3)
                products.append({
                    "sku": f"SKU-{n:04d}",
                    "name": f"{brand} {adj} {noun}",
                    "category": category,
                    "price": round(rng.uniform(lo, hi), 2),
                    "in_stock": rng.random() < 0.9,
                    "description": f"{adj} {noun} by {brand}: {', '.join(feats)}.",
                })
    return products


def build_orders(catalog: list[dict], n_orders: int = 300, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    statuses = ["delivered"] * 50 + ["shipped"] * 20 + ["processing"] * 15 + ["cancelled"] * 5 + ["returned"] * 10
    orders: list[dict] = []
    for i in range(n_orders):
        status = rng.choice(statuses)
        items = [{"sku": p["sku"], "qty": rng.randint(1, 3)} for p in rng.sample(catalog, rng.randint(1, 3))]
        if status in ("delivered", "returned"):
            days_ago = rng.randint(8, 60)
        elif status == "cancelled":
            days_ago = rng.randint(1, 20)
        else:
            days_ago = rng.randint(1, 6)
        order_date = REFERENCE_TODAY - timedelta(days=days_ago)
        order = {
            "order_id": f"WM-{10001 + i}",
            "status": status,
            "order_date": order_date.isoformat(),
            "items": items,
            "delivered_date": None,
            "eta": None,
            "carrier": None,
            "tracking_number": None,
        }
        if status in ("delivered", "returned"):
            order["delivered_date"] = (order_date + timedelta(days=rng.randint(2, 6))).isoformat()
            order["carrier"] = rng.choice(["UPS", "FedEx", "USPS"])
        elif status == "shipped":
            order["carrier"] = rng.choice(["UPS", "FedEx", "USPS"])
            order["tracking_number"] = f"TRK{rng.randint(10**9, 10**10 - 1)}"
            order["eta"] = (REFERENCE_TODAY + timedelta(days=rng.randint(1, 4))).isoformat()
        orders.append(order)
    return orders


def write_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    orders = build_orders(catalog)
    (DATA_DIR / "catalog.json").write_text(json.dumps(catalog, indent=1))
    (DATA_DIR / "orders.json").write_text(json.dumps(orders, indent=1))
    print(f"wrote {len(catalog)} products and {len(orders)} orders to {DATA_DIR}")


def load_catalog() -> list[dict]:
    path = DATA_DIR / "catalog.json"
    if not path.exists():
        return build_catalog()
    return json.loads(path.read_text())


def load_orders() -> list[dict]:
    path = DATA_DIR / "orders.json"
    if not path.exists():
        return build_orders(load_catalog())
    return json.loads(path.read_text())


if __name__ == "__main__":
    write_all()
