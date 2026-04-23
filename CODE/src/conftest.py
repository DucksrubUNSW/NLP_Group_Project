#!/usr/bin/env python3

# helper for test_headline_predictions.py

import pytest

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)

def pytest_sessionfinish(session, exitstatus):
    from test_headline_predictions import _results, RESULTS_FILE
    with open(RESULTS_FILE, "w") as f:
        for line in _results:
            f.write(line + "\n")
        passed = sum(1 for r in _results if r.startswith("PASSED"))
        total = len(_results)
        f.write(f"\n{passed}/{total} passed\n")
    print(f"\nResults saved to {RESULTS_FILE}")
