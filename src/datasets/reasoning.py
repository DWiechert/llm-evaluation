"""reasoning category: same numeric-extraction approach as finance — these are riddles
with one unambiguous computed answer, not open-ended judgment calls."""

# "All but 9 die" means 9 survive — the answer is stated in the question itself.
SHEEP_LEFT = 9.0
# $84 / 3 = $28 each originally. Remove the mistaken $12 item: $72 / 3 = $24 each.
# The question asks how much LESS that is: 28 - 24.
BILL_SAVINGS_PER_PERSON = 4.0
# 8 balls, one heavier, balance scale: ceil(log_3(8)) = 2 weighings (3^2 = 9 >= 8).
MIN_WEIGHINGS = 2.0

REASONING_DATASET = [
    {
        "inputs": {"id": "reasoning_1", "category": "reasoning", "messages": [
            {"role": "user", "content": (
                "A farmer has 17 sheep. All but 9 die. How many sheep does the "
                "farmer have left? Explain your reasoning step by step."
            )}
        ]},
        "expectations": {"expected_number": SHEEP_LEFT, "tolerance": 0.5},
    },
    {
        "inputs": {"id": "reasoning_2", "category": "reasoning", "messages": [
            {"role": "user", "content": (
                "Three friends split a restaurant bill of $84 evenly. Then they "
                "realize a $12 item was mistakenly added to the bill that none of "
                "them ordered. After removing it and splitting the remainder "
                "evenly, how much does each person owe, and how much less is "
                "that than their original share?"
            )}
        ]},
        "expectations": {"expected_number": BILL_SAVINGS_PER_PERSON, "tolerance": 0.5},
    },
    {
        "inputs": {"id": "reasoning_3", "category": "reasoning", "messages": [
            {"role": "user", "content": (
                "You have 8 identical-looking balls, one of which is slightly "
                "heavier than the rest. Using a balance scale, describe the "
                "minimum number of weighings needed to identify the heavier "
                "ball, and explain the strategy."
            )}
        ]},
        "expectations": {"expected_number": MIN_WEIGHINGS, "tolerance": 0.5},
    },
]
