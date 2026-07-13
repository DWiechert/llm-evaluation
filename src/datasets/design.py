"""design category: NO deterministic scorer exists for open-ended system design quality.
These are included for side-by-side manual review only. The scorer attached to this
category is a coverage heuristic (does it mention expected concerns), not a quality
judgment — treat it as a checklist, not a grade."""

DESIGN_DATASET = [
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
