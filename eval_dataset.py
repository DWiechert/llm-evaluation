"""
eval_dataset.py — Evaluation data for mlflow_eval.py

Every category here is scored with rule-based/code scorers only (no LLM judge,
per your preference). That means:

  - basic_questions: exact/substring match against a known-correct short answer
  - tool_usage: checks the model picked the right tool + required args (structural)
  - coding: ACTUALLY RUNS the generated code against test cases — real pass/fail
  - finance: extracts the model's numeric answer and checks it against a value
             computed with plain Python (see the compute block below), not guessed
  - reasoning: same numeric-extraction approach as finance — these are riddles
             with one unambiguous computed answer, not open-ended judgment calls
  - instruction_following: checks concrete, machine-verifiable constraints from
             the prompt (sentence/bullet counts, word limits, forbidden words,
             required JSON keys) — not prose quality
  - design: NO deterministic scorer exists for open-ended system design quality.
             These are included for side-by-side manual review only. The scorer
             attached to this category is a coverage heuristic (does it mention
             expected concerns), not a quality judgment — treat it as a checklist,
             not a grade.
"""

# ---- Finance expected values, computed directly (see comment for the math) ----
# A = P(1+r/n)^(nt); P=10000, r=0.06, n=12, t=3
COMPOUND_INTEREST_ANSWER = 11966.81
# ROI: invest 5000 -> 6750 after 2 years
TOTAL_ROI_PCT = 35.0
CAGR_PCT = 16.19
# Break-even: fixed=12000, price=45, variable=20 -> fixed/(price-variable)
BREAKEVEN_UNITS = 480.0

# ---- Reasoning expected values, computed directly ----
# "All but 9 die" means 9 survive — the answer is stated in the question itself.
SHEEP_LEFT = 9.0
# $84 / 3 = $28 each originally. Remove the mistaken $12 item: $72 / 3 = $24 each.
# The question asks how much LESS that is: 28 - 24.
BILL_SAVINGS_PER_PERSON = 4.0
# 8 balls, one heavier, balance scale: ceil(log_3(8)) = 2 weighings (3^2 = 9 >= 8).
MIN_WEIGHINGS = 2.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["title", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

EVAL_DATASET = [
    # ---------------- basic_questions ----------------
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

    # ---------------- tool_usage ----------------
    {
        "inputs": {"id": "tool_usage_1", "category": "tool_usage", "tools": TOOLS, "messages": [
            {"role": "user", "content": "What's the current price of Apple stock?"}
        ]},
        "expectations": {"expected_tool": "get_stock_price", "required_args": ["symbol"]},
    },
    {
        "inputs": {"id": "tool_usage_2", "category": "tool_usage", "tools": TOOLS, "messages": [
            {"role": "user", "content": "Convert 250 US dollars to Japanese yen."}
        ]},
        "expectations": {"expected_tool": "convert_currency", "required_args": ["amount", "from_currency", "to_currency"]},
    },
    {
        "inputs": {"id": "tool_usage_3", "category": "tool_usage", "tools": TOOLS, "messages": [
            {"role": "user", "content": "Set up a meeting called 'Sprint Planning' on 2026-07-10 at 10:00."}
        ]},
        "expectations": {"expected_tool": "create_calendar_event", "required_args": ["title", "date"]},
    },

    # ---------------- coding (executed, real pass/fail) ----------------
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

    # ---------------- finance (numeric, computed not guessed) ----------------
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

    # ---------------- reasoning (numeric, computed not guessed) ----------------
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

    # ---------------- instruction_following (structural, machine-checked) ----
    {
        "inputs": {"id": "instruction_following_1", "category": "instruction_following", "messages": [
            {"role": "user", "content": (
                "Write a 3-sentence product description for a wireless mechanical "
                "keyboard. Sentence 1 must mention battery life. Sentence 2 must "
                "mention switch type. Sentence 3 must be a call to action. Do not "
                "use the word 'amazing' or any exclamation points."
            )}
        ]},
        "expectations": {
            "sentence_count": 3,
            "required_substrings_by_sentence": {"0": ["battery"], "1": ["switch"]},
            "cta_sentence_index": 2,
            "cta_keywords": ["buy", "order", "get", "shop", "grab", "upgrade", "try", "add to cart"],
            "forbidden_substrings": ["amazing", "!"],
        },
    },
    {
        "inputs": {"id": "instruction_following_2", "category": "instruction_following", "messages": [
            {"role": "user", "content": (
                "Summarize the plot of a heist movie in exactly 5 bullet points, "
                "each bullet no more than 12 words."
            )}
        ]},
        "expectations": {"bullet_count": 5, "max_words_per_bullet": 12},
    },
    {
        "inputs": {"id": "instruction_following_3", "category": "instruction_following", "messages": [
            {"role": "user", "content": (
                "Respond to this message ONLY in valid JSON with keys 'sentiment' "
                "(positive/negative/neutral) and 'reason' (string, max 20 words): "
                "'The new update broke my workflow but support fixed it within an "
                "hour.'"
            )}
        ]},
        "expectations": {
            "json_required_keys": ["sentiment", "reason"],
            "sentiment_options": ["positive", "negative", "neutral"],
            "reason_max_words": 20,
        },
    },

    # ---------------- design (manual review — no ground truth) ----------------
    {
        "inputs": {"id": "design_1", "category": "design", "messages": [
            {"role": "user", "content": (
                "Design a system to ingest and store 10 million IoT sensor readings per day, "
                "each with a timestamp, sensor ID, and float value. Readers need both recent "
                "(last 24h) and historical (multi-year) queries. Outline your architecture and "
                "key tradeoffs."
            )}
        ]},
        "expectations": {"review_only": True, "expected_concepts": ["time-series", "partition", "retention", "index"]},
    },
    {
        "inputs": {"id": "design_2", "category": "design", "messages": [
            {"role": "user", "content": (
                "You're designing the data model for GrowthCanvas, a 2.5D isometric garden "
                "planner. Users place plants on a grid of garden beds and need to see planting "
                "history over time. Propose a schema and explain your key design decisions."
            )}
        ]},
        "expectations": {"review_only": True, "expected_concepts": ["foreign key", "history", "grid", "coordinate"]},
    },
]
