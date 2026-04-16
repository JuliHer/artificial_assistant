import yaml

with open("src/configs/intentions.yaml", "r") as f:
    INTENT_MAP = yaml.safe_load(f)