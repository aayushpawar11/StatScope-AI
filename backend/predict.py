from data import get_latest_stats

SUPPORTED_STATS = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk"
}

def predict_stat(player, stat, threshold):
    if stat not in SUPPORTED_STATS:
        return {"error": f"Stat '{stat}' not supported. Try one of: {list(SUPPORTED_STATS.keys())}"}

    stats = get_latest_stats(player)
    if not stats:
        return {"error": "Player not found or no recent games"}

    stat_key = SUPPORTED_STATS[stat]
    values = [game[stat_key] for game in stats if stat_key in game]
    if not values:
        return {"error": f"No recent data found for stat '{stat}'"}

    avg = sum(values) / len(values)
    prediction = "Yes" if avg > threshold else "No"
    confidence = f"{min(99, int(abs(avg - threshold) / threshold * 100))}%"

    return {
        "player": player,
        "stat": stat,
        "threshold": threshold,
        "average_last_5": round(avg, 2),
        "prediction": prediction,
        "confidence": confidence
    }
