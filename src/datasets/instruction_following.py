"""instruction_following category: checks concrete, machine-verifiable constraints from
the prompt (sentence/bullet counts, word limits, forbidden words, required JSON keys) —
not prose quality."""

INSTRUCTION_DATASET = [
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
]
