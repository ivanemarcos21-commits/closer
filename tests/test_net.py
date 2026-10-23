"""Tests for closer.net: certificate detection and the app-wide search budget."""

from __future__ import annotations

import ssl
import threading

import httpx

from closer.net import SearchBudget, is_certificate_error


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestCertificateDetection:
    def test_finds_a_certificate_error_buried_in_the_cause_chain(self):
        outer = httpx.ConnectError("boom")
        outer.__cause__ = RuntimeError("wrapper")
        outer.__cause__.__cause__ = ssl.SSLCertVerificationError("self-signed certificate")
        assert is_certificate_error(outer)

    def test_ordinary_errors_are_not_certificate_errors(self):
        assert not is_certificate_error(httpx.ConnectTimeout("slow"))
        assert not is_certificate_error(None)

    def test_a_cause_loop_cannot_hang_it(self):
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__cause__, b.__cause__ = b, a
        assert not is_certificate_error(a)


class TestSearchBudget:
    def test_allows_up_to_the_cap_then_refuses(self):
        budget = SearchBudget(3, clock=Clock())
        assert [budget.try_acquire() for _ in range(5)] == [True, True, True, False, False]

    def test_old_searches_stop_counting_after_an_hour(self):
        clock = Clock()
        budget = SearchBudget(2, clock=clock)
        assert budget.try_acquire() and budget.try_acquire() and not budget.try_acquire()
        clock.now += 3599
        assert not budget.try_acquire()
        clock.now += 2
        assert budget.try_acquire()

    def test_a_refused_search_does_not_use_up_budget(self):
        clock = Clock()
        budget = SearchBudget(1, clock=clock)
        assert budget.try_acquire()
        assert not budget.try_acquire() and not budget.try_acquire()
        clock.now += 3600
        assert budget.try_acquire() and not budget.try_acquire()

    def test_is_safe_when_many_visitors_search_at_once(self):
        budget = SearchBudget(25, clock=Clock())
        granted: list[bool] = []
        lock = threading.Lock()

        def visitor() -> None:
            result = budget.try_acquire()
            with lock:
                granted.append(result)

        threads = [threading.Thread(target=visitor) for _ in range(60)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert granted.count(True) == 25 and granted.count(False) == 35

    def test_a_cap_of_zero_still_allows_one(self):
        assert SearchBudget(0, clock=Clock()).try_acquire()
