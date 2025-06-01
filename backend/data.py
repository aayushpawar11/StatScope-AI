import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("BALLDONTLIE_API_KEY")

BASE_URL = "https://www.balldontlie.io/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
} if API_KEY else {}

# Get player ID from name
def get_player_id(name):
    try:
        response = requests.get(
            f"{BASE_URL}/players",
            params={"search": name},
            headers=HEADERS
        )
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        if data["data"]:
            return data["data"][0]["id"]
        else:
            print(f"No player found for: {name}")
            return None
    except Exception as e:
        print(f"Exception in get_player_id: {e}")
        return None

# Get latest 5 games' stats for player
def get_latest_stats(player_name):
    player_id = get_player_id(player_name)
    if not player_id:
        return None

    try:
        response = requests.get(
            f"{BASE_URL}/stats",
            params={"player_ids[]": player_id, "per_page": 5},
            headers=HEADERS
        )
        if response.status_code != 200:
            print(f"⚠️ Stats API Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        return data["data"] if "data" in data else None
    except Exception as e:
        print(f"Exception in get_latest_stats: {e}")
        return None
