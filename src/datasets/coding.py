"""coding category: ACTUALLY RUNS the generated code against test cases — real pass/fail."""

CODING_DATASET = [
    {
        "inputs": {"id": "coding_1", "category": "coding", "messages": [
            {"role": "user", "content": (
                "Write a Python function `second_largest_unique(nums)` that returns the second "
                "largest unique value in a list of integers. Raise ValueError if fewer than 2 "
                "unique values exist. Respond with ONLY the code, no explanation, no markdown fences."
            )}
        ]},
        "expectations": {
            "function_name": "second_largest_unique",
            "test_cases": [
                {"args": [[1, 2, 3, 4, 5]], "expected": 4},
                {"args": [[5, 5, 5, 3]], "expected": 3},
                {"args": [[10, 10, 10]], "expected": "ValueError"},
            ],
        },
    },
    {
        "inputs": {"id": "coding_2", "category": "coding", "messages": [
            {"role": "user", "content": (
                "Write a Python function `is_balanced(s)` that returns True if a string of "
                "brackets ()[]{} is balanced, False otherwise. Respond with ONLY the code, "
                "no explanation, no markdown fences."
            )}
        ]},
        "expectations": {
            "function_name": "is_balanced",
            "test_cases": [
                {"args": ["({[]})"], "expected": True},
                {"args": ["([)]"], "expected": False},
                {"args": [""], "expected": True},
            ],
        },
    },
]
