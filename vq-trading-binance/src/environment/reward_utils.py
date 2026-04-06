def calculate_profit_risk_reward(profit, risk, weight=0.5):
    """
    Reward = profit - (risk * weight)
    rt = profit - risk
    """
    return profit - (risk * weight)
