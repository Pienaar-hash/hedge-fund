from utils.firestore_client import get_db
import os

def _doc(name:str):
    env = os.environ.get("ENV","dev")
    db = get_db()
    return db.collection("hedge").document(env).collection("state").document(name).get()

def read_leaderboard(env=None):
    return _doc("leaderboard").to_dict() or {}

def read_nav(env=None):
    return _doc("nav").to_dict() or {}

def read_positions(env=None):
    return _doc("positions").to_dict() or {}
