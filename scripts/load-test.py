#!/usr/bin/env python3
"""
scripts/load-test.py — Load test to verify horizontal scaling.

Milestone 28: Test the backend with concurrent users.

What this tests:
  1. Concurrent chat requests (agentic workflow under load)
  2. Latency distribution (p50, p95, p99)
  3. Error rate under load
  4. Whether HPA scales the backend pods

How to run:
  # Start the backend first
  cd enterprise-ai-assistant
  source venv/bin/activate
  uvicorn backend.app.main:app --port 8000

  # Run load test
  python scripts/load-test.py --url http://localhost:8000 --users 10 --duration 60

  # Against Kubernetes
  python scripts/load-test.py --url http://api.yourdomain.com --users 50 --duration 120
"""

import asyncio
import argparse
import statistics
import time
import json
from typing import List
import aiohttp


async def send_chat_request(session: aiohttp.ClientSession, url: str, user_id: str) -> dict:
    """Send a single chat request and return timing info."""
    payload = {
        "user_id": user_id,
        "session_id": f"load-test-{user_id}",
        "message": "What is RAG and how does it work?",
        "stream": False,
    }

    start = time.time()
    status = 0
    error = None

    try:
        async with session.post(
            f"{url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            status = response.status
            if status == 200:
                await response.json()
            elif status == 429:
                error = "rate_limited"
            else:
                error = f"http_{status}"

    except asyncio.TimeoutError:
        error = "timeout"
        status = 0
    except Exception as e:
        error = str(e)
        status = 0

    latency_ms = (time.time() - start) * 1000
    return {
        "status": status,
        "latency_ms": latency_ms,
        "error": error,
        "user_id": user_id,
    }


async def run_user(session: aiohttp.ClientSession, url: str, user_id: str, duration_seconds: int) -> List[dict]:
    """Simulate one user sending requests for the given duration."""
    results = []
    end_time = time.time() + duration_seconds

    while time.time() < end_time:
        result = await send_chat_request(session, url, user_id)
        results.append(result)
        await asyncio.sleep(2)   # Think time between requests

    return results


async def run_load_test(url: str, num_users: int, duration_seconds: int):
    """Run concurrent load test with multiple simulated users."""
    print(f"\n{'='*60}")
    print(f" Load Test: {url}")
    print(f" Users: {num_users} | Duration: {duration_seconds}s")
    print(f"{'='*60}\n")

    # Check if backend is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print("✅ Backend is reachable")
                else:
                    print(f"⚠️  Backend returned {resp.status}")
    except Exception as e:
        print(f"❌ Backend not reachable: {e}")
        return

    print(f"\n Starting {num_users} concurrent users...")

    all_results = []
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            run_user(session, url, f"user_{i}", duration_seconds)
            for i in range(num_users)
        ]
        user_results = await asyncio.gather(*tasks)

    for results in user_results:
        all_results.extend(results)

    total_time = time.time() - start_time

    # ── Results Analysis ──────────────────────────────────────────────────────
    if not all_results:
        print("No results!")
        return

    total_requests = len(all_results)
    successful = [r for r in all_results if r["status"] == 200]
    failed = [r for r in all_results if r["error"]]
    rate_limited = [r for r in all_results if r.get("error") == "rate_limited"]
    timeouts = [r for r in all_results if r.get("error") == "timeout"]

    latencies = [r["latency_ms"] for r in successful]

    print(f"\n{'='*60}")
    print(f" Load Test Results")
    print(f"{'='*60}")
    print(f"\n📊 Request Statistics:")
    print(f"   Total requests:   {total_requests}")
    print(f"   Successful (200): {len(successful)} ({len(successful)/total_requests*100:.1f}%)")
    print(f"   Failed:           {len(failed)} ({len(failed)/total_requests*100:.1f}%)")
    print(f"   Rate limited:     {len(rate_limited)}")
    print(f"   Timeouts:         {len(timeouts)}")
    print(f"   RPS:              {total_requests/total_time:.1f} requests/second")

    if latencies:
        print(f"\n⏱️  Latency Distribution (successful requests):")
        print(f"   Min:   {min(latencies):.0f}ms")
        print(f"   p50:   {statistics.median(latencies):.0f}ms")
        print(f"   p95:   {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
        print(f"   p99:   {statistics.quantiles(latencies, n=100)[98]:.0f}ms")
        print(f"   Max:   {max(latencies):.0f}ms")
        print(f"   Mean:  {statistics.mean(latencies):.0f}ms")

    print(f"\n✅ Load test complete! Total time: {total_time:.1f}s")

    # ── Scaling recommendation ────────────────────────────────────────────────
    median_latency = statistics.median(latencies) if latencies else 0
    if median_latency > 5000:
        print(f"\n⚠️  High latency (p50={median_latency:.0f}ms). Consider:")
        print(f"   - Scaling: kubectl scale deployment/backend --replicas=5 -n ai-assistant")
        print(f"   - Check: kubectl get hpa -n ai-assistant")
    elif len(failed) / total_requests > 0.1:
        print(f"\n⚠️  High error rate ({len(failed)/total_requests*100:.1f}%). Check backend logs.")
    else:
        print(f"\n🎉 System is handling load well!")


def main():
    parser = argparse.ArgumentParser(description="Load test the Enterprise AI Assistant")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--users", type=int, default=5, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.users, args.duration))


if __name__ == "__main__":
    main()
