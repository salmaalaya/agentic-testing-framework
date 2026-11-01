"""
target_app/knowledge_base.py — Base de connaissances factuelle TechShop

Extrait de target_app/app.py pour être importable sans déclencher la
construction de l'app FastAPI (templates, StaticFiles, routes...). Consommé
par :
  - target_app/app.py       : moteur de réponse (PRODUCTS, ORDERS)
  - src/agentic_testing/evaluation/metrics.py : contexte factuel pour le score NLI
    (détection d'hallucination par entailment, cf. compute_nli_score())
"""

# Catalogue FICTIF (TechShop n'existe pas) — prix en Dinars Tunisiens (DT),
# choisis pour rester réalistes par rapport au marché pour des produits comparables.
PRODUCTS = {
    "laptop-pro": {
        "name": "LaptopPro X1",
        "price": 1299.99,
        "stock": 15,
        "description": "Laptop 15 pouces, Intel Core i7, 16GB RAM, 512GB SSD",
        "category": "informatique",
        "keywords": ["laptop", "ordinateur", "pc", "computer", "computadora", "ноутбук", "حاسوب", "컴퓨터", "电脑"],
    },
    "phone-ultra": {
        "name": "PhoneUltra 5G",
        "price": 899.99,
        "stock": 42,
        "description": "Smartphone 6.7 pouces, 128GB, caméra 108MP",
        "category": "mobile",
        "keywords": ["phone", "smartphone", "téléphone", "mobile", "celular", "телефон", "هاتف", "휴대폰", "手机"],
    },
    "headset-pro": {
        "name": "HeadsetPro ANC",
        "price": 249.99,
        "stock": 0,
        "description": "Casque audio Bluetooth, réduction de bruit active",
        "category": "audio",
        "keywords": ["casque", "headset", "headphones", "auriculares", "наушники", "سماعة", "헤드셋", "耳机"],
    },
    "tablet-air": {
        "name": "TabletAir 10",
        "price": 899.00,
        "stock": 22,
        "description": "Tablette 10.9 pouces, 128GB, écran Full HD, stylet inclus",
        "category": "informatique",
        "keywords": ["tablette", "tablet", "ipad", "tableta", "планшет", "لوحي", "태블릿", "平板"],
    },
    "earbuds-x": {
        "name": "EarBudsX TWS",
        "price": 89.90,
        "stock": 60,
        "description": "Écouteurs sans fil Bluetooth 5.3, boîtier de charge, 25h d'autonomie",
        "category": "audio",
        "keywords": [
            "écouteurs",
            "earbuds",
            "ecouteur",
            "airpods",
            "tws",
            "auriculares inalambricos",
            "سماعات لاسلكية",
            "이어버드",
        ],
    },
    "gamerbook-rtx": {
        "name": "GamerBook RTX",
        "price": 3819.00,
        "stock": 6,
        "description": "PC portable gamer 16 pouces, RTX 5060, Intel Core i7, 16GB RAM, 1TB SSD",
        "category": "informatique",
        "keywords": ["gamer", "gaming", "rtx", "pc gamer", "portable gamer", "juegos", "игровой", "قيمنق"],
    },
}

ORDERS = {
    "1234": {
        "status": "En livraison",
        "date": "18 juillet 2026",
        "estimated": "22 juillet 2026",
        "product": "LaptopPro X1",
    },
    "5678": {"status": "Livré", "date": "10 juillet 2026", "delivered": "13 juillet 2026", "product": "PhoneUltra 5G"},
    "9999": {
        "status": "En préparation",
        "date": "19 juillet 2026",
        "estimated": "24 juillet 2026",
        "product": "HeadsetPro ANC",
    },
}


def build_factual_context() -> str:
    """
    Construit un texte court résumant les faits connus (catalogue + commandes)
    — utilisé comme prémisse NLI pour vérifier si la réponse de l'agent
    (hypothèse) est cohérente (entailment), neutre, ou contredit ces faits
    (contradiction = hallucination probable).
    """
    lines = []
    for product in PRODUCTS.values():
        stock_txt = f"{product['stock']} unités en stock" if product["stock"] > 0 else "en rupture de stock"
        lines.append(f"{product['name']} coûte {product['price']} DT et est {stock_txt}.")
    for order_id, order in ORDERS.items():
        lines.append(f"La commande #{order_id} ({order['product']}) est au statut : {order['status']}.")
    return " ".join(lines)
