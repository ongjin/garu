"""도메인별 rate limiter. 도메인 단위 최소 간격 강제."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from rate_limiter import RateLimiter

def test_first_request_no_wait():
    rl = RateLimiter(min_interval_sec=1.0)
    t0 = time.monotonic()
    rl.wait("example.com")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, f"first request shouldn't wait, but waited {elapsed}s"

def test_second_request_to_same_host_waits():
    rl = RateLimiter(min_interval_sec=0.3)
    rl.wait("example.com")
    t0 = time.monotonic()
    rl.wait("example.com")
    elapsed = time.monotonic() - t0
    assert 0.25 <= elapsed <= 0.5, f"expected ~0.3s wait, got {elapsed}s"

def test_different_hosts_independent():
    rl = RateLimiter(min_interval_sec=0.3)
    rl.wait("a.com")
    t0 = time.monotonic()
    rl.wait("b.com")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, f"different hosts shouldn't share limit, waited {elapsed}s"

def test_per_host_override():
    rl = RateLimiter(min_interval_sec=1.0, per_host_overrides={"slow.com": 2.0})
    rl.wait("slow.com")
    t0 = time.monotonic()
    rl.wait("slow.com")
    elapsed = time.monotonic() - t0
    assert 1.9 <= elapsed <= 2.2, f"expected ~2.0s for slow.com, got {elapsed}s"

if __name__ == "__main__":
    test_first_request_no_wait(); print("first_no_wait OK")
    test_second_request_to_same_host_waits(); print("same_host_wait OK")
    test_different_hosts_independent(); print("different_hosts OK")
    test_per_host_override(); print("override OK")
    print("RATE LIMITER OK")
