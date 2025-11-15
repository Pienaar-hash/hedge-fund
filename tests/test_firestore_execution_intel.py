from execution.firestore_utils import publish_execution_intel


def test_publish_execution_intel_is_noop(monkeypatch):
    called = False

    def fake_client():
        nonlocal called
        called = True
        raise AssertionError("Firestore client should not be used")

    monkeypatch.setattr("execution.firestore_utils._direct_client", fake_client)

    publish_execution_intel("BTCUSDC", {"score": 1.0})
    assert called is False
