import time
import random
import logging
from execution.firestore_utils import publish_heartbeat
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
while True:
    try:
        publish_heartbeat(service="sync_state", status="ok")
        logging.info("[heartbeat] shim published ok")
    except Exception as e:
        logging.warning(f"[heartbeat] shim failed: {e}")
    time.sleep(random.uniform(5,10))
