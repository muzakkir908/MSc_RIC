#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fog Cache Server (True LRU + optional TTL, robust cloud calls, optional latency normalization)

- Receives Edge predictions and local context
- Discretises state to make a compact key
- True LRU cache for state->decision (thread-safe) + optional TTL
- On cache miss, calls Cloud /decide with retries and stores the result
- Exposes /process, /stats, /health

Expected payload (Edge -> Fog):
{
  "edge_prediction": {
    "prediction": 1,
    "probability": 0.85,
    "confidence": 0.92,
    "latency_ms": 0
  },
  "current_latency": 65.0,  # ms, unless normalization enabled
  "network_quality": 0.8,   # 0..1
  "cpu_load": 0.75,         # 0..1
  "time_since_combat": 0.0  # 0..1 or seconds (you decide)
}
"""

import os
import time
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional, List

import requests
from flask import Flask, request, jsonify

# ---------------------------
# Config
# ---------------------------

# Where the Cloud service is listening
CLOUD_HOST = os.getenv("CLOUD_HOST", "127.0.0.1")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "5001"))
CLOUD_DECIDE_URL = f"http://{CLOUD_HOST}:{CLOUD_PORT}/decide"

# Flask
FOG_HOST = os.getenv("FOG_HOST", "0.0.0.0")
FOG_PORT = int(os.getenv("FOG_PORT", "5002"))
DEBUG = os.getenv("FOG_DEBUG", "0") == "1"

# LRU Cache settings
CACHE_CAPACITY = int(os.getenv("FOG_CACHE_CAPACITY", "512"))
# TTL in seconds; set to None, "", "0", "None" to disable expiry
_cache_ttl_env = os.getenv("FOG_CACHE_TTL", "2.0")
CACHE_TTL: Optional[float] = None if _cache_ttl_env in ("", "0", "None", "none") else float(_cache_ttl_env)

# Latency normalization
NORMALIZE_LATENCY = os.getenv("FOG_NORMALIZE_LATENCY", "1") in ("1", "true", "True", "yes", "Y")
MAX_LAT_MS = float(os.getenv("FOG_MAX_LAT_MS", "200"))  # used if NORMALIZE_LATENCY=1

# If not normalizing, you can override latency bins (ms) via env: e.g. "0,50,100,200,inf"
LATENCY_BINS_MS_ENV = os.getenv("FOG_LATENCY_BINS_MS", "0,50,100,200,inf")

# Discretisation bins (match your Q-learning). These are default 0..1 bins.
DEFAULT_BINS: Dict[str, List[float]] = {
    "combat_probability": [0.0, 0.3, 0.7, 1.0],   # -> 0,1,2
    "latency":            [0.0, 0.33, 0.67, 1.0], # -> 0,1,2 (if normalized)
    "network_quality":    [0.0, 0.5, 1.0],        # -> 0,1
    "cpu_load":           [0.0, 0.5, 1.0],        # -> 0,1
    "time_since_combat":  [0.0, 0.33, 0.67, 1.0]  # -> 0,1,2
}

def _parse_ms_bins(env_str: str) -> List[float]:
    parts = [p.strip() for p in env_str.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        if p.lower() in ("inf", "infinity", "+inf"):
            out.append(float("inf"))
        else:
            out.append(float(p))
    # Ensure strictly increasing
    if len(out) < 2:
        out = [0.0, float("inf")]
    return out

# Build final BINS, possibly substituting latency if not normalizing
if NORMALIZE_LATENCY:
    BINS = DEFAULT_BINS.copy()
else:
    # Replace with ms bins
    BINS = DEFAULT_BINS.copy()
    BINS["latency"] = _parse_ms_bins(LATENCY_BINS_MS_ENV)

# Cloud call settings
CLOUD_TIMEOUT = float(os.getenv("FOG_CLOUD_TIMEOUT", "2.5"))  # seconds
CLOUD_RETRIES = int(os.getenv("FOG_CLOUD_RETRIES", "2"))
CLOUD_BACKOFF = float(os.getenv("FOG_CLOUD_BACKOFF", "0.2"))  # seconds between retries

# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fog")

# ---------------------------
# True LRU Cache (thread-safe, optional TTL)
# ---------------------------

class LRUCache:
    def __init__(self, capacity: int = 512, ttl: Optional[float] = None):
        self.capacity = capacity
        self.ttl = ttl  # seconds, or None for no TTL
        self.store: "OrderedDict[str, Tuple[Dict[str, Any], float]]" = OrderedDict()
        self._lock = threading.Lock()

    def _expired(self, ts: float) -> bool:
        return (self.ttl is not None) and ((time.time() - ts) > self.ttl)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self.store:
                return None
            value, ts = self.store[key]
            if self._expired(ts):
                # drop expired
                del self.store[key]
                return None
            # mark as recently used
            self.store.move_to_end(key, last=True)
            return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            if key in self.store:
                self.store.move_to_end(key, last=True)
            self.store[key] = (value, now)
            # evict least-recently used
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self.store)

# Instantiate cache & stats
cache = LRUCache(capacity=CACHE_CAPACITY, ttl=CACHE_TTL)
cache_hits = 0
cache_misses = 0
_counter_lock = threading.Lock()

# ---------------------------
# Helpers
# ---------------------------

def _bin_value(x: float, edges: List[float]) -> int:
    """
    Return the index bin for value x given monotonic edges.
    Example edges [0.0,0.3,0.7,1.0] -> bins 0..2 for intervals
    """
    # Clamp weird input
    if x is None:
        x = 0.0
    try:
        xf = float(x)
    except Exception:
        xf = 0.0

    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        # Handle infinite right edge gracefully
        if left <= xf < right:
            return i

    # If equals max edge or beyond, put in last bin-1
    return max(0, len(edges) - 2)

def _normalize_latency(lat_ms: float) -> float:
    """Normalize latency ms to [0,1] using MAX_LAT_MS cap."""
    try:
        v = float(lat_ms)
    except Exception:
        v = 0.0
    if MAX_LAT_MS <= 0:
        return 0.0
    return max(0.0, min(1.0, v / MAX_LAT_MS))

def discretise_state(payload: Dict[str, Any]) -> Dict[str, int]:
    """
    Make bin indices for the 5 state features.
    Expects:
      payload["edge_prediction"]["probability"]
      payload["current_latency"]
      payload["network_quality"]
      payload["cpu_load"]
      payload["time_since_combat"]
    """
    ep = payload.get("edge_prediction", {}) or {}
    combat_prob = ep.get("probability", 0.0)

    latency_raw = payload.get("current_latency", 0.0)
    latency_input = _normalize_latency(latency_raw) if NORMALIZE_LATENCY else float(latency_raw)

    netq = payload.get("network_quality", 0.0)
    cpu = payload.get("cpu_load", 0.0)
    tsc = payload.get("time_since_combat", 0.0)

    return {
        "combat_bin": _bin_value(combat_prob, BINS["combat_probability"]),
        "latency_bin": _bin_value(latency_input, BINS["latency"]),
        "network_quality_bin": _bin_value(netq, BINS["network_quality"]),
        "cpu_load_bin": _bin_value(cpu, BINS["cpu_load"]),
        "time_since_combat_bin": _bin_value(tsc, BINS["time_since_combat"]),
    }

def make_state_key(bins: Dict[str, int]) -> str:
    # Stable ordering for key construction
    return f"{bins['combat_bin']}:{bins['latency_bin']}:{bins['network_quality_bin']}:{bins['cpu_load_bin']}:{bins['time_since_combat_bin']}"

def call_cloud_decide(state_bins: Dict[str, int]) -> Dict[str, Any]:
    """
    POST to Cloud /decide with a compact state (bins) and return JSON decision.
    Retries on network or bad-response issues.
    """
    req = {"state_bins": state_bins}  # match your cloud_server /decide contract
    last_err: Optional[str] = None

    for attempt in range(1, CLOUD_RETRIES + 1):
        try:
            resp = requests.post(CLOUD_DECIDE_URL, json=req, timeout=CLOUD_TIMEOUT)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                raise requests.RequestException("Invalid JSON from Cloud")
            return data
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning("Cloud /decide attempt %d/%d failed: %s", attempt, CLOUD_RETRIES, e)
            if attempt < CLOUD_RETRIES:
                time.sleep(CLOUD_BACKOFF)

    log.error("Cloud /decide failed after %d attempts: %s", CLOUD_RETRIES, last_err)
    # Fallback decision if Cloud unreachable or invalid
    return {
        "action": "medium",
        "slice": "medium",
        "expected_latency": 60.0,
        "reason": "cloud_unreachable_fallback",
        "state": state_bins,
    }

# ---------------------------
# Flask App
# ---------------------------

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    """
    Receives Edge->Fog payload, discretises state, uses LRU cache for decision.
    On miss, calls Cloud /decide and caches the decision.
    """
    global cache_hits, cache_misses

    payload = request.get_json(silent=True) or {}
    if "edge_prediction" not in payload:
        return jsonify({"status": "error", "error": "missing edge_prediction"}), 400

    # Discretise to bins
    bins = discretise_state(payload)
    state_key = make_state_key(bins)

    # LRU cache lookup
    decision = cache.get(state_key)
    if decision is not None:
        with _counter_lock:
            cache_hits += 1
        from_cache = True
    else:
        decision = call_cloud_decide(bins)
        cache.set(state_key, decision)
        with _counter_lock:
            cache_misses += 1
        from_cache = False

    # Optional: include edge prob in response for debugging
    edge_prob = payload.get("edge_prediction", {}).get("probability", None)

    return jsonify({
        "status": "ok",
        "from_cache": from_cache,
        "state_bins": bins,
        "edge_probability": edge_prob,
        "decision": decision
    }), 200

@app.route("/stats", methods=["GET"])
def stats():
    with _counter_lock:
        hits = cache_hits
        misses = cache_misses
    total = hits + misses
    hit_rate = (hits / total) if total else 0.0
    return jsonify({
        "status": "ok",
        "cache": {
            "type": "LRU",
            "capacity": CACHE_CAPACITY,
            "size": len(cache),
            "ttl": CACHE_TTL,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 3)
        },
        "cloud": {
            "url": CLOUD_DECIDE_URL,
            "timeout_s": CLOUD_TIMEOUT,
            "retries": CLOUD_RETRIES,
        },
        "latency": {
            "normalized": NORMALIZE_LATENCY,
            "max_ms": MAX_LAT_MS if NORMALIZE_LATENCY else None,
            "bins": BINS["latency"],
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    log.info(
        "Starting Fog server on %s:%s (LRU capacity=%d, ttl=%s, normalize_latency=%s, max_lat_ms=%.1f) -> Cloud %s (timeout=%.1fs, retries=%d)",
        FOG_HOST, FOG_PORT, CACHE_CAPACITY, CACHE_TTL, NORMALIZE_LATENCY, MAX_LAT_MS,
        CLOUD_DECIDE_URL, CLOUD_TIMEOUT, CLOUD_RETRIES
    )
    # threaded=True is fine; we added thread-safety to cache & counters
    app.run(host=FOG_HOST, port=FOG_PORT, debug=DEBUG, threaded=True)
