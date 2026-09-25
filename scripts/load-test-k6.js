/**
 * load-test-k6.js — Production load test for the Enterprise AI Assistant.
 *
 * WHAT IS k6?
 *   k6 is a developer-focused load testing tool (like Apache JMeter but modern).
 *   It runs this JavaScript file and simulates virtual users (VUs) hitting your API.
 *
 * WHY LOAD TEST?
 *   Your app might work fine with 1 user but break with 100 simultaneous users.
 *   Load testing reveals:
 *   - When response time becomes unacceptable (> 2s for chat is bad UX)
 *   - When the server runs out of database connections
 *   - When Redis rate limiting kicks in and blocks requests
 *   - When you need more Kubernetes pods (autoscaling threshold)
 *
 * SCENARIOS:
 *   1. health_check  — warm up, confirm servers are up
 *   2. chat_load     — simulate concurrent chat users (main scenario)
 *   3. document_load — simulate concurrent document uploads
 *   4. rag_stress    — high-concurrency RAG queries (most expensive operation)
 *
 * STAGES (gradual ramp-up):
 *   0→10 VUs  → 2 min   (warm up)
 *   10→50 VUs → 3 min   (normal load)
 *   50→100 VUs→ 3 min   (high load)
 *   100 VUs   → 5 min   (sustained high load)
 *   100→0 VUs → 2 min   (ramp down)
 *
 * INSTALL k6:
 *   # Ubuntu/Debian:
 *   sudo gpg -k
 *   sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
 *     --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
 *   echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
 *     | sudo tee /etc/apt/sources.list.d/k6.list
 *   sudo apt update && sudo apt install k6
 *
 * RUN:
 *   # Quick smoke (10 VUs, 1 min):
 *   k6 run --vus 10 --duration 1m scripts/load-test-k6.js
 *
 *   # Full ramp-up test:
 *   k6 run scripts/load-test-k6.js
 *
 *   # Target specific environment:
 *   BASE_URL=http://api.prod.example.com k6 run scripts/load-test-k6.js
 *
 *   # Output to JSON for Grafana:
 *   k6 run --out json=results.json scripts/load-test-k6.js
 *
 * EXPECTED OUTPUT:
 *   ✓ http_req_duration............: avg=450ms  p(90)=1200ms  p(95)=1800ms
 *   ✓ http_req_failed..............: 0.12%
 *   ✓ checks.......................: 98.5%
 */

import http from "k6/http";
import { check, sleep, group } from "k6";
import { Rate, Trend, Counter } from "k6/metrics";

// ── Custom metrics ─────────────────────────────────────────────────────────
// These appear in k6 output alongside standard metrics.
const chatErrorRate = new Rate("chat_errors");       // fraction of failed chat requests
const chatDuration = new Trend("chat_duration_ms");   // response time distribution
const documentErrors = new Rate("document_errors");   // fraction of failed uploads
const ragLatency = new Trend("rag_latency_ms");        // RAG-specific latency

// ── Configuration ──────────────────────────────────────────────────────────
const BACKEND_URL = __ENV.BASE_URL || "http://localhost:8000";
const LANGGRAPH_URL = __ENV.LG_URL || "http://localhost:7860";

// ── Load stages ────────────────────────────────────────────────────────────
// VU = Virtual User (one simulated user session)
export const options = {
  stages: [
    { duration: "1m",  target: 5  },   // Stage 1: Warm up (5 VUs)
    { duration: "2m",  target: 20 },   // Stage 2: Normal load (20 VUs)
    { duration: "3m",  target: 50 },   // Stage 3: High load (50 VUs)
    { duration: "3m",  target: 100 },  // Stage 4: Sustained high load
    { duration: "2m",  target: 0  },   // Stage 5: Ramp down
  ],

  // ── Thresholds ────────────────────────────────────────────────────────
  // The test FAILS if any threshold is breached.
  // These become your SLOs (Service Level Objectives).
  thresholds: {
    // 95% of chat requests must complete within 3 seconds
    "http_req_duration{scenario:chat}": ["p(95)<3000"],

    // 99% of health checks must be < 200ms (fast health check = responsive system)
    "http_req_duration{scenario:health}": ["p(99)<200"],

    // Error rate must stay below 1%
    "http_req_failed": ["rate<0.01"],

    // Custom: chat-specific errors < 2%
    "chat_errors": ["rate<0.02"],

    // Custom: RAG p95 < 5 seconds (RAG is slower due to embedding + search)
    "rag_latency_ms": ["p(95)<5000"],
  },

  // ── Scenario tags (for Grafana filtering) ─────────────────────────────
  tags: {
    app: "enterprise-ai-assistant",
    environment: __ENV.ENVIRONMENT || "local",
  },
};

// ── Shared headers ─────────────────────────────────────────────────────────
const JSON_HEADERS = {
  "Content-Type": "application/json",
  Accept: "application/json",
};

// ── Helper: generate unique session IDs ────────────────────────────────────
function sessionId() {
  return `k6-session-${__VU}-${Date.now()}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO 1: Health Checks
// Runs first to confirm servers are alive before chat load.
// ─────────────────────────────────────────────────────────────────────────────
export function healthCheck() {
  group("health_checks", () => {
    // Backend health
    const backendRes = http.get(`${BACKEND_URL}/health`, {
      tags: { scenario: "health", service: "backend" },
    });
    check(backendRes, {
      "backend: status is 200": (r) => r.status === 200,
      "backend: has status field": (r) => {
        try { return JSON.parse(r.body).status !== undefined; }
        catch { return false; }
      },
    });

    // LangGraph health
    const lgRes = http.get(`${LANGGRAPH_URL}/api/health`, {
      tags: { scenario: "health", service: "langgraph" },
    });
    check(lgRes, {
      "langgraph: status is 200": (r) => r.status === 200,
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO 2: Chat Load
// Simulates N users simultaneously sending chat messages.
// ─────────────────────────────────────────────────────────────────────────────
const SAMPLE_QUERIES = [
  "What are the key technical skills mentioned in the resume?",
  "What is the candidate's work experience?",
  "List the educational qualifications.",
  "What programming languages does the candidate know?",
  "What projects has the candidate worked on?",
  "Summarise the candidate's strengths.",
  "What certifications does the candidate hold?",
  "How many years of experience does the candidate have?",
];

export function chatLoad() {
  const query = SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
  const session = sessionId();

  group("chat_backend", () => {
    const startTime = Date.now();
    const res = http.post(
      `${BACKEND_URL}/api/chat`,
      JSON.stringify({ session_id: session, message: query }),
      { headers: JSON_HEADERS, tags: { scenario: "chat", service: "backend" } }
    );

    const duration = Date.now() - startTime;
    chatDuration.add(duration);

    const ok = check(res, {
      "chat: status 200": (r) => r.status === 200,
      "chat: has response": (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.message || body.response || body.answer;
        } catch { return false; }
      },
      "chat: duration < 5s": () => duration < 5000,
    });

    chatErrorRate.add(!ok);
  });

  // Simulate human reading time between messages (1–3 seconds)
  sleep(1 + Math.random() * 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO 3: LangGraph Chat Load
// Tests the LangGraph agent (7-node graph) separately.
// ─────────────────────────────────────────────────────────────────────────────
export function langraphChatLoad() {
  const query = SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];

  group("chat_langgraph", () => {
    const startTime = Date.now();
    const res = http.post(
      `${LANGGRAPH_URL}/api/chat`,
      JSON.stringify({
        query: query,
        user_id: `k6-vu-${__VU}`,
      }),
      {
        headers: JSON_HEADERS,
        tags: { scenario: "chat", service: "langgraph" },
        timeout: "10s",  // LangGraph can be slow due to 7-node graph
      }
    );

    const duration = Date.now() - startTime;
    ragLatency.add(duration);

    check(res, {
      "lg-chat: not 5xx": (r) => r.status < 500,
      "lg-chat: duration < 10s": () => duration < 10000,
    });
  });

  sleep(2 + Math.random() * 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO 4: Document List (read-heavy)
// Most users only READ documents, not upload them.
// ─────────────────────────────────────────────────────────────────────────────
export function documentRead() {
  group("document_read", () => {
    const res = http.get(`${BACKEND_URL}/api/documents`, {
      headers: JSON_HEADERS,
      tags: { scenario: "document_read" },
    });

    check(res, {
      "doc-list: status 200": (r) => r.status === 200,
      "doc-list: is array or object": (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body) || typeof body === "object";
        } catch { return false; }
      },
    });
  });

  sleep(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT SCENARIO (runs when no --env is specified)
// Mix of all scenarios in realistic proportions:
//   70% chat queries (most common user action)
//   20% document reads
//   10% health checks
// ─────────────────────────────────────────────────────────────────────────────
export default function () {
  const rand = Math.random();

  if (rand < 0.70) {
    chatLoad();
  } else if (rand < 0.90) {
    documentRead();
  } else {
    healthCheck();
  }
}

// ── Setup: runs once before the test starts ────────────────────────────────
export function setup() {
  console.log("═══════════════════════════════════════════════");
  console.log("  Enterprise AI Assistant — Load Test");
  console.log(`  Backend:  ${BACKEND_URL}`);
  console.log(`  LangGraph: ${LANGGRAPH_URL}`);
  console.log("═══════════════════════════════════════════════");

  // Pre-flight health check
  const r = http.get(`${BACKEND_URL}/health`);
  if (r.status !== 200) {
    console.error(`WARN: Backend health check failed (${r.status}). Test may produce errors.`);
  }

  return {
    backendUrl: BACKEND_URL,
    lgUrl: LANGGRAPH_URL,
    startTime: new Date().toISOString(),
  };
}

// ── Teardown: runs once after the test ends ────────────────────────────────
export function teardown(data) {
  console.log("═══════════════════════════════════════════════");
  console.log("  Load test complete.");
  console.log(`  Started:  ${data.startTime}`);
  console.log(`  Finished: ${new Date().toISOString()}`);
  console.log("═══════════════════════════════════════════════");
}
