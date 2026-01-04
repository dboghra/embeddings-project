import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # reads TRELLO_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID from .env

TRELLO_KEY = os.getenv("TRELLO_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID")
OUT_PATH = Path("data")
OUT_PATH.mkdir(exist_ok=True)

def fetch_board_cards(key: str, token: str, board_id: str):
    url = f"https://api.trello.com/1/boards/{board_id}/cards"
    params = {
        "key": key,
        "token": token,
        # fields commonly useful for analysis
        "fields": "id,name,desc,labels,idList,due,shortUrl,dateLastActivity",
        "attachments": "true",
        "attachment_fields": "url,name",
        "checklists": "all",
        "members": "true",
        "member_fields": "fullName",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def main():
    if not (TRELLO_KEY and TRELLO_TOKEN and TRELLO_BOARD_ID):
        raise SystemExit("Missing TRELLO_KEY/TRELLO_TOKEN/TRELLO_BOARD_ID in environment (.env).")
    cards = fetch_board_cards(TRELLO_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID)
    out_file = OUT_PATH / "cards.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cards)} cards to {out_file}")

if __name__ == "__main__":
    main()
