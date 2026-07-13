"""basic_questions category: exact/substring match against a known-correct short answer."""

BASIC_DATASET = [
    {
        "inputs": {"id": "basic_questions_1", "category": "basic_questions", "messages": [
            {"role": "user", "content": "What is the time complexity of binary search on a sorted array of n elements?"}
        ]},
        "expectations": {"expected_substring": "log n"},
    },
    {
        "inputs": {"id": "basic_questions_2", "category": "basic_questions", "messages": [
            {"role": "user", "content": "How many bits are in a byte?"}
        ]},
        "expectations": {"expected_substring": "8"},
    },
    {
        "inputs": {"id": "basic_questions_3", "category": "basic_questions", "messages": [
            {"role": "user", "content": "In HTTP, what status code indicates a resource was not found?"}
        ]},
        "expectations": {"expected_substring": "404"},
    },
]
