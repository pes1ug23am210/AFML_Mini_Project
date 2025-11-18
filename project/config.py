# config.py
import yaml

# Load the data.yaml configuration
with open('data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

# Get class names from the config
class_names = {int(k): v for k, v in data_config['names'].items()}
num_classes = data_config['nc']

print(f"Processing {num_classes} classes: {list(class_names.values())}")

# ---------------------------------------------------------------------------
# Simple price list per product class for billing.
# You can adjust these prices to match your real products.
# Keys must match the class names in data.yaml (values of data_config['names']).
# ---------------------------------------------------------------------------
DEFAULT_PRICE = 50.0  # fallback price if a class is missing from the table
PRODUCT_PRICES = {
    'axe-perfume': 250.0,
    'cinthol-soap': 40.0,
    'comfort': 90.0,
    'dettol': 80.0,
    'garnier-face-cream': 180.0,
    'head-and-shoulder': 120.0,
    'himalya-face-wash': 120.0,
    'loreal': 220.0,
    'maggie': 30.0,
    'nivea-body-lotion': 200.0,
    'ponds-cream': 150.0,
    'savlon-handwash': 75.0,
    'sprite': 40.0,
    'wild-stone-powder': 160.0,
    'wild-stone-soap': 70.0,
}

def get_product_price(class_name: str) -> float:
    """Return the price for a given product class name.

    If the class is not explicitly listed, fall back to DEFAULT_PRICE.
    """
    return float(PRODUCT_PRICES.get(class_name, DEFAULT_PRICE))