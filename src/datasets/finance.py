"""finance category: extracts the model's numeric answer and checks it against a value
computed with plain Python (see the compute block below), not guessed."""

# A = P(1+r/n)^(nt); P=10000, r=0.06, n=12, t=3
COMPOUND_INTEREST_ANSWER = 11966.81
# ROI: invest 5000 -> 6750 after 2 years
TOTAL_ROI_PCT = 35.0
CAGR_PCT = 16.19
# Break-even: fixed=12000, price=45, variable=20 -> fixed/(price-variable)
BREAKEVEN_UNITS = 480.0

FINANCE_DATASET = [
    {
        "inputs": {"id": "finance_1", "category": "finance", "messages": [
            {"role": "user", "content": (
                "You invest $10,000 at an annual interest rate of 6%, compounded monthly, "
                "for 3 years. What is the final amount? Give just the dollar figure to the cent."
            )}
        ]},
        "expectations": {"expected_number": COMPOUND_INTEREST_ANSWER, "tolerance": 1.0},
    },
    {
        "inputs": {"id": "finance_2", "category": "finance", "messages": [
            {"role": "user", "content": (
                "I invested $5,000 and sold the investment for $6,750 two years later. "
                "What is my total ROI as a percentage?"
            )}
        ]},
        "expectations": {"expected_number": TOTAL_ROI_PCT, "tolerance": 0.5},
    },
    {
        "inputs": {"id": "finance_3", "category": "finance", "messages": [
            {"role": "user", "content": (
                "A product has fixed costs of $12,000, sells for $45/unit, and costs $20/unit "
                "to produce. How many units must be sold to break even?"
            )}
        ]},
        "expectations": {"expected_number": BREAKEVEN_UNITS, "tolerance": 1.0},
    },
]
