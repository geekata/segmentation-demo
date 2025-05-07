import requests
from datetime import datetime
from pathlib import Path

BASE_URL = "http://microscope.local:5000"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def get_last_capture():
    try:
        response = requests.get(f"{BASE_URL}/api/v2/captures")
        response.raise_for_status()
        captures = response.json()

        if not captures:
            raise ValueError("No captures found")

        # Find the most recent capture by time
        newest = max(captures, key=lambda x: datetime.fromisoformat(x["time"]))
        return newest
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to get captures: {e}")
    except ValueError as e:
        raise Exception(f"Error: {e}")

def download_capture(capture):
    try:
        capture_id = capture["id"]
        filename = capture["name"]
        download_url = f"{BASE_URL}/api/v2/captures/{capture_id}/download/{filename}"

        response = requests.get(download_url)
        response.raise_for_status()

        path = DATA_DIR / filename
        with open(path, "wb") as f:
            f.write(response.content)

        return path
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download capture: {e}")
