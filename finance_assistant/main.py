def process_message(message: str) -> dict:
    """
    Router Agent: Classifies intent and routes to the correct sub-agent.
    Returns a dictionary matching the React Native frontend schema.
    """
    message_lower = message.lower()
    
    # Base response template
    response = {
        "ai_reply": "I am your multi-agent router. How can I assist you?",
        "total_balance": 10950.00,
        "budget_status": "On Track",
        "category_breakdown": [],
        "upcoming_bills": [],
        "portfolio_health": None
    }

    # Route 1: Bills & Subscriptions Agent
    if "bill" in message_lower or "subscription" in message_lower:
        response["ai_reply"] = "Here are your upcoming bills for this week."
        response["upcoming_bills"] = [
            {"bill_name": "Electricity", "due_date": "2026-10-20", "amount": 85.00},
            {"bill_name": "Spotify", "due_date": "2026-10-22", "amount": 10.99}
        ]
        
    # Route 2: Expenditure & Analyst Agent
    elif "budget" in message_lower or "spend" in message_lower:
        response["ai_reply"] = "Here is your spending breakdown for this month."
        response["budget_status"] = "Warning: High Food Spend"
        response["category_breakdown"] = [
            {"category": "Housing", "amount": 1200},
            {"category": "Food", "amount": 600},
            {"category": "Transport", "amount": 150}
        ]
        
    # Route 3: Investment & Portfolio Agent
    elif "invest" in message_lower or "portfolio" in message_lower:
        response["ai_reply"] = "Your portfolio is currently outperforming the market by 2%."
        response["portfolio_health"] = "Bullish (+2.4%)"
        
    # Default Route: General Chat
    else:
        response["ai_reply"] = f"Router received: '{message}'. Which specific area (bills, budget, investments) would you like to explore?"

    return response