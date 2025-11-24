from __future__ import annotations

import types

import pytest
import requests

import execution.exchange_utils as ex


def test_classify_binance_error_transient_and_auth():
    resp_rate = requests.Response()
    resp_rate.status_code = 429
    resp_rate._content = b'{"code": -1003, "msg": "Too many requests"}'  # type: ignore[attr-defined]
    classification = ex.classify_binance_error(requests.HTTPError(response=resp_rate), response=resp_rate)
    assert classification["category"] == "rate_limit"
    assert classification["retriable"] is True

    resp_auth = requests.Response()
    resp_auth.status_code = 401
    resp_auth._content = b'{"code": -4061, "msg": "API-key format invalid."}'  # type: ignore[attr-defined]
    classification_auth = ex.classify_binance_error(requests.HTTPError(response=resp_auth), response=resp_auth)
    assert classification_auth["category"] == "auth"
    assert classification_auth["retriable"] is False


def test_req_retries_on_transient_requesterror(monkeypatch):
    calls: list[int] = []

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    def fake_request(method, url, data=None, timeout=None, headers=None):
        calls.append(1)
        if len(calls) == 1:
            raise requests.RequestException("network glitch")
        return DummyResponse()

    monkeypatch.setattr(ex, "_S", types.SimpleNamespace(request=fake_request))
    monkeypatch.setattr(ex, "_BACKOFF_INITIAL", 0.0)
    monkeypatch.setattr(ex, "_BACKOFF_MAX", 0.0)
    monkeypatch.setattr(ex.time, "sleep", lambda _: None)

    resp = ex._req("GET", "/fapi/v1/ping")
    assert isinstance(resp, DummyResponse)
    assert len(calls) == 2


def test_classify_binance_client_error_like():
    class DummyClientError(Exception):
        def __init__(self, status_code, error_code, message):
            super().__init__(message)
            self.status_code = status_code
            self.error_code = error_code
            self.error_message = message

    rate_err = DummyClientError(429, -1003, "Too many requests")
    rate_classification = ex.classify_binance_error(rate_err)
    assert rate_classification["category"] == "rate_limit"
    assert rate_classification["retriable"] is True
    assert rate_classification["status"] == 429
    assert rate_classification["code"] == -1003

    auth_err = DummyClientError(400, -2015, "Invalid API-key")
    auth_classification = ex.classify_binance_error(auth_err)
    assert auth_classification["category"] == "auth"
    assert auth_classification["retriable"] is False
    assert auth_classification["status"] == 400
    assert auth_classification["code"] == -2015
