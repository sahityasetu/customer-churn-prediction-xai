"""Turn churn probabilities into a retention-campaign decision.

All money values are ASSUMPTIONS for illustration. Change them in one place here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CampaignAssumptions:
    offer_cost: float = 50.0        # cost of one retention offer (discount / call), in $
    save_rate: float = 0.30         # share of real churners who stay after getting the offer
    value_months: int = 12          # months of revenue we keep when a churner is saved


def expected_profit(y_true, proba, monthly_charges, threshold, a: CampaignAssumptions = CampaignAssumptions()):
    """Profit of contacting every customer with proba >= threshold."""
    y_true = np.asarray(y_true)
    target = np.asarray(proba) >= threshold
    saved_value = (target & (y_true == 1)) * np.asarray(monthly_charges) * a.value_months * a.save_rate
    return float(saved_value.sum() - target.sum() * a.offer_cost)


def best_threshold(y_true, proba, monthly_charges, a: CampaignAssumptions = CampaignAssumptions()):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = np.array([expected_profit(y_true, proba, monthly_charges, t, a) for t in grid])
    return float(grid[profits.argmax()]), grid, profits
