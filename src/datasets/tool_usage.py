"""tool_usage category: checks the model picked the right tool + required args (structural)."""

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

TOOL_DATASET = [
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
]
