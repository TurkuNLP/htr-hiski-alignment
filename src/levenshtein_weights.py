import json
from functools import cache
from pathlib import Path

import numpy as np

from src.set_costs import (
    set_cost,
    set_cost_from_codes,
    set_cost_two_way,
    set_cost_two_way_from_codes,
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


@cache
def _read_spec(path:Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _codes(chars: str, size: int) -> list:
    codes = [ord(c) for c in chars]
    too_large = [c for c in codes if c >= size]
    if too_large:
        raise ValueError(f"character codes {too_large} are not below size {size}")
    return codes


def _build_char_costs(spec: dict, size: int) -> np.ndarray:
    costs = np.full(size, fill_value=spec["default"], dtype=np.float64)
    for char, cost in spec["overrides"].items():
        _codes(char, size)
        set_cost(costs, cost, char)
    return costs


def _build_weights(spec: dict) -> dict:
    size = spec["size"]
    classes = {
        name: _codes(chars, size) for name, chars in spec["classes"].items()
    }

    insert_costs = _build_char_costs(spec["insert"], size)
    delete_costs = _build_char_costs(spec["delete"], size)

    sub = spec["substitute"]
    substitute_costs = np.full(
        (size, size), fill_value=sub["default"], dtype=np.float64
    )
    for cost, class_a, class_b, two_way in sub["class_rules"]:
        setter = (
            set_cost_two_way_from_codes if two_way else set_cost_from_codes
        )
        setter(substitute_costs, cost, classes[class_a], classes[class_b])

    np.fill_diagonal(substitute_costs, sub["self_cost"])

    for cost, char_a, char_b, two_way in sub["pair_rules"]:
        codes_a = _codes(char_a, size)
        codes_b = _codes(char_b, size)
        if two_way:
            set_cost_two_way(substitute_costs, cost, char_a, char_b)
        else:
            set_cost_from_codes(substitute_costs, cost, codes_a, codes_b)

    deabbreviate_spec = spec["deabbreviate"]
    deabbreviate_costs = {}
    for symbol, entries in deabbreviate_spec["symbols"].items():
        costs = np.full(
            size, fill_value=deabbreviate_spec["default"], dtype=np.float64
        )
        for cost, class_names in entries:
            codes = [c for name in class_names for c in classes[name]]
            set_cost_from_codes(costs, cost, codes)
        deabbreviate_costs[symbol] = costs

    return {
        "insert_costs": insert_costs,
        "delete_costs": delete_costs,
        "substitute_costs": substitute_costs,
        "deabbreviate_costs": deabbreviate_costs,
    }


def get_predefined_weights(path:Path) -> dict:
    return _build_weights(_read_spec(path))