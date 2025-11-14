from execution.firestore_utils import publish_execution_intel


class DummyDoc:
    def __init__(self):
        self.set_calls = []
        self.collections = {}

    def set(self, data, merge=False):
        self.set_calls.append((data, merge))

    def collection(self, name):
        coll = self.collections.setdefault(name, DummyCollection())
        return coll


class DummyCollection:
    def __init__(self):
        self.docs = {}

    def document(self, doc_id):
        doc = self.docs.setdefault(doc_id, DummyDoc())
        return doc


class DummyClient:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        coll = self.collections.setdefault(name, DummyCollection())
        return coll


def test_publish_execution_intel_writes_document_execution_intelligence(monkeypatch):
    client = DummyClient()

    monkeypatch.setattr("execution.firestore_utils._direct_client", lambda: client)
    monkeypatch.setenv("HEDGE_ENV", "test")

    payload = {"score": 1.0, "components": {"dummy": 1}}
    publish_execution_intel("BTCUSDC", payload)

    hedge_coll = client.collection("hedge")
    env_doc = hedge_coll.document("test")
    intel_doc = env_doc.collection("execution_intel").document("BTCUSDC")
    assert intel_doc.set_calls, "execution_intel doc should be written"
    data, merge = intel_doc.set_calls[-1]
    assert data["symbol"] == "BTCUSDC"
    assert data["score"] == 1.0
    assert merge is True
