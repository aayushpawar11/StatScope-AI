from data import get_latest_stats
from model import predict_over_under

SUPPORTED_STATS = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk"
}

def predict_stat(player, stat, threshold):
    if stat not in SUPPORTED_STATS:
        return {"error": f"Stat '{stat}' not supported."}

    stats = get_latest_stats(player)
    if not stats:
        return {"error": "Player not found or no recent games"}

    # Use last game (for now)
    latest_game = stats[0]
    input_stats = [
        latest_game.get("pts", 0),
        latest_game.get("reb", 0),
        latest_game.get("ast", 0),
        latest_game.get("stl", 0),
        latest_game.get("blk", 0),
        latest_game.get("min", 0)
    ]

    result = predict_over_under(input_stats, threshold)
    result.update({
        "player": player,
        "stat": stat,
        "threshold": threshold,
        "input_stats": input_stats
    })
    return result
