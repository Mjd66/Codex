#!/usr/bin/env python3
"""
Codex Unified v5
Single-file autonomous engineering agent with:
- OpenAI Responses API support (with Chat Completions fallback)
- planner / builder / reviewer loop
- sqlite memory with WAL + FTS5 retrieval
- safe workspace tools
- resilient web search + fetch tools
- regex patching, lint/test helpers
- security scan helpers
- task graph for longer-horizon work
- simulations and curated module catalogs
- GeeksforGeeks-compatible AI module ideas catalog
- CLI entrypoint

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

VERSION = "codex_unified_v5"
DEFAULT_USER_AGENT = f"{VERSION}/1.0"
GFG_SOURCE_URL = "https://www.geeksforgeeks.org/artificial-intelligence/best-artificial-intelligence-project-ideas/"


# ============================================================
# Core helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def fetch_text_url(url: str, timeout: int = 20, max_chars: int = 16000) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return body[:max_chars]


# ============================================================
# Config and models
# ============================================================

@dataclass(slots=True)
class AgentConfig:
    model: str = "gpt-5"
    planner_model: str | None = None
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    workspace: Path = Path.cwd()
    memory_db_path: Path = Path(".codex_memory.sqlite3")
    max_turns: int = 16
    temperature: float = 0.2
    max_tool_output_chars: int = 16000
    request_timeout: int = 180
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5
    reviewer_every: int = 4
    use_responses_api: bool = True
    verbose: bool = False
    writing_speed: float = 1.2
    activity_level: float = 1.25
    base_energy: float = 120.0
    heat_resistance: float = 0.85
    burn_rate: float = 8.0
    heat_absorb_rate: float = 0.5
    reminiscence_depth: int = 8
    memory_max_rows: int = 5000
    resilience_factor: float = 1.0
    negative_forgetting_rate: float = 0.2

    @classmethod
    def from_env(cls) -> "AgentConfig":
        workspace = Path(os.getenv("CODEX_WORKSPACE", str(Path.cwd()))).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        memory_db_path = Path(
            os.getenv("CODEX_MEMORY_DB", str(workspace / ".codex_memory.sqlite3"))
        ).resolve()
        memory_db_path.parent.mkdir(parents=True, exist_ok=True)

        return cls(
            model=os.getenv("CODEX_MODEL", "gpt-5"),
            planner_model=os.getenv("CODEX_PLANNER_MODEL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            workspace=workspace,
            memory_db_path=memory_db_path,
            max_turns=int(os.getenv("CODEX_MAX_TURNS", "16")),
            temperature=float(os.getenv("CODEX_TEMPERATURE", "0.2")),
            max_tool_output_chars=int(os.getenv("CODEX_MAX_TOOL_OUTPUT", "16000")),
            request_timeout=int(os.getenv("CODEX_REQUEST_TIMEOUT", "180")),
            retry_attempts=int(os.getenv("CODEX_RETRY_ATTEMPTS", "3")),
            retry_backoff_seconds=float(os.getenv("CODEX_RETRY_BACKOFF", "1.5")),
            reviewer_every=int(os.getenv("CODEX_REVIEWER_EVERY", "4")),
            use_responses_api=os.getenv("CODEX_USE_RESPONSES_API", "1").lower() not in {"0", "false", "no"},
            verbose=os.getenv("CODEX_VERBOSE", "0").lower() in {"1", "true", "yes"},
            writing_speed=float(os.getenv("CODEX_WRITING_SPEED", "1.2")),
            activity_level=float(os.getenv("CODEX_ACTIVITY_LEVEL", "1.25")),
            base_energy=float(os.getenv("CODEX_BASE_ENERGY", "120")),
            heat_resistance=float(os.getenv("CODEX_HEAT_RESISTANCE", "0.85")),
            burn_rate=float(os.getenv("CODEX_BURN_RATE", "8")),
            heat_absorb_rate=float(os.getenv("CODEX_HEAT_ABSORB_RATE", "0.5")),
            reminiscence_depth=int(os.getenv("CODEX_REMINISCENCE_DEPTH", "8")),
            memory_max_rows=int(os.getenv("CODEX_MEMORY_MAX_ROWS", "5000")),
            resilience_factor=float(os.getenv("CODEX_RESILIENCE_FACTOR", "1.0")),
            negative_forgetting_rate=float(os.getenv("CODEX_NEGATIVE_FORGETTING_RATE", "0.2")),
        )


@dataclass(slots=True)
class MemoryItem:
    key: str
    value: str
    tags: str = ""
    created_at: str = field(default_factory=utc_now)
    importance: float = 0.5
    last_accessed: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ToolCall:
    id: str
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelTurn:
    content: str
    tool_calls: list[ToolCall]
    raw: dict[str, Any]


@dataclass(slots=True)
class AgentRunResult:
    final_answer: str
    turns_used: int
    tool_calls: int
    task_summary: str


@dataclass(slots=True)
class PerformanceState:
    writing_speed: float
    activity_level: float
    energy: float
    heat_resistance: float
    burn_rate: float
    heat_absorb_rate: float
    heat: float = 0.0

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def burn_to_heat(self) -> float:
        burn = self.burn_rate * self.activity_level * self.writing_speed
        self.energy = max(0.0, self.energy - burn)
        heat_gain = burn * max(0.15, 1.0 - self.heat_resistance)
        self.heat += heat_gain
        return burn

    def absorb_heat_to_energy(self) -> float:
        absorb = self.heat * self.heat_absorb_rate
        self.heat = max(0.0, self.heat - absorb)
        self.energy = self._clamp(self.energy + absorb, 0.0, 300.0)
        return absorb

    def passive_cooling(self) -> float:
        cooled = min(self.heat, 1.5 + (self.heat_resistance * 2.0))
        self.heat = max(0.0, self.heat - cooled)
        return cooled

    def tune_for_turn(self, base_temperature: float, turn: int, max_turns: int) -> float:
        self.burn_to_heat()
        self.absorb_heat_to_energy()
        self.passive_cooling()

        progress = turn / max(1, max_turns)
        activity_boost = 0.08 * (self.activity_level - 1.0)
        energy_boost = 0.1 if self.energy > 90 else 0.0
        heat_drag = 0.14 if self.heat > 40 else 0.0
        finisher_push = 0.05 if progress > 0.65 else 0.0
        tuned = base_temperature + activity_boost + energy_boost + finisher_push - heat_drag
        return self._clamp(tuned, 0.0, 1.2)

    def snapshot(self) -> str:
        return (
            f"speed={self.writing_speed:.2f}, activity={self.activity_level:.2f}, "
            f"energy={self.energy:.1f}, heat={self.heat:.1f}, "
            f"heat_resistance={self.heat_resistance:.2f}, absorb={self.heat_absorb_rate:.2f}"
        )


@dataclass(slots=True)
class FutureModule:
    name: str
    description: str
    weight: float
    heat_cost: float
    energy_gain: float
    stability_gain: float
    innovation_gain: float


@dataclass(slots=True)
class SimulationResult:
    steps: int
    final_energy: float
    final_heat: float
    avg_temperature: float
    stability_index: float
    speed_index: float
    innovation_index: float
    module_scores: dict[str, float]


@dataclass(slots=True)
class CompatibleIdea:
    title: str
    category: str
    benefit: str
    implementation_hint: str
    source_url: str = GFG_SOURCE_URL


# ============================================================
# Curated module catalogs
# ============================================================

def build_geeksforgeeks_compatible_catalog() -> list[CompatibleIdea]:
    return [
        CompatibleIdea(
            title="Chatbots",
            category="agent interface",
            benefit="Improves conversational frontend and niche assistant workflows.",
            implementation_hint="Add domain-routed dialogue modes and command parsing modules.",
        ),
        CompatibleIdea(
            title="Fake News Detection System",
            category="content verification",
            benefit="Useful as a trust filter for retrieved articles and external text.",
            implementation_hint="Add a lightweight credibility scoring layer before storing web memories.",
        ),
        CompatibleIdea(
            title="Sentiment Predictor",
            category="state analysis",
            benefit="Can estimate emotional tone of user text or model output for adaptive style control.",
            implementation_hint="Add lexicon-based fallback sentiment scoring in standard library mode.",
        ),
        CompatibleIdea(
            title="Object Detector",
            category="multimodal extension",
            benefit="Future-compatible bridge for image or VR scene understanding.",
            implementation_hint="Represent as a placeholder module spec and integration stub for external CV runtimes.",
        ),
        CompatibleIdea(
            title="Recommender Engine",
            category="memory ranking",
            benefit="Improves retrieval ranking for memories, tools, and next actions.",
            implementation_hint="Blend recency, importance, overlap, and usage into recommendation scores.",
        ),
        CompatibleIdea(
            title="Image Caption Generator",
            category="multimodal extension",
            benefit="Useful future module for describing screenshots, diagrams, and VR frames.",
            implementation_hint="Keep as catalog metadata plus integration notes for external models.",
        ),
        CompatibleIdea(
            title="Detecting Spam Emails",
            category="message hygiene",
            benefit="Can filter noisy or malicious text before the agent spends tokens on it.",
            implementation_hint="Add heuristic spam and prompt-injection phrase detection.",
        ),
        CompatibleIdea(
            title="Language Translation",
            category="language support",
            benefit="Expands the agent to multilingual tasks and memory normalization.",
            implementation_hint="Add translation-module placeholders and language routing hooks.",
        ),
        CompatibleIdea(
            title="Text Summarization",
            category="compression",
            benefit="Perfect fit for summarizing tool output, fetched pages, and long files.",
            implementation_hint="Add local sentence-ranking summarizer and structured summary memory entries.",
        ),
        CompatibleIdea(
            title="Text Autocorrector",
            category="input repair",
            benefit="Can repair noisy user prompts and malformed JSON/tool arguments.",
            implementation_hint="Add typo-aware normalization before strict parsing.",
        ),
        CompatibleIdea(
            title="Personalized Voice Assistant",
            category="interface extension",
            benefit="Useful future voice mode for command workflows.",
            implementation_hint="Expose voice assistant as a planned module instead of embedding non-stdlib dependencies.",
        ),
        CompatibleIdea(
            title="Inventory Demand Forecasting",
            category="forecasting",
            benefit="Can inspire resource forecasting for memory growth, tool usage, and task load.",
            implementation_hint="Add simple rolling forecast utilities for internal metrics.",
        ),
        CompatibleIdea(
            title="Speech Recognition",
            category="interface extension",
            benefit="Natural future bridge for spoken commands.",
            implementation_hint="Keep as catalog idea with placeholders for later external integration.",
        ),
        CompatibleIdea(
            title="Reviews Analysis",
            category="feedback loop",
            benefit="Useful for mining strengths and weaknesses from user feedback.",
            implementation_hint="Map feedback to action items and memory importance scores.",
        ),
    ]


def select_compatible_geeksforgeeks_modules(limit: int = 8) -> list[CompatibleIdea]:
    preferred = {
        "agent interface",
        "content verification",
        "state analysis",
        "memory ranking",
        "compression",
        "input repair",
        "message hygiene",
        "forecasting",
        "feedback loop",
    }
    items = [item for item in build_geeksforgeeks_compatible_catalog() if item.category in preferred]
    return items[:limit]


def build_future_modules() -> list[FutureModule]:
    return [
        FutureModule(
            name="usc_consensus",
            description="Universal self-consistency style consensus for selecting best candidate outputs.",
            weight=1.05,
            heat_cost=0.9,
            energy_gain=0.6,
            stability_gain=1.2,
            innovation_gain=0.6,
        ),
        FutureModule(
            name="self_discover_reasoner",
            description="Self-discover task-specific reasoning structures before heavy execution.",
            weight=1.0,
            heat_cost=1.1,
            energy_gain=0.5,
            stability_gain=0.9,
            innovation_gain=1.3,
        ),
        FutureModule(
            name="evolutionary_builder",
            description="Iterative mutate-score-select loop for code and plans.",
            weight=1.15,
            heat_cost=1.4,
            energy_gain=0.8,
            stability_gain=0.7,
            innovation_gain=1.6,
        ),
        FutureModule(
            name="summary_compressor",
            description="Text summarization inspired compression module for long tool outputs.",
            weight=1.0,
            heat_cost=0.7,
            energy_gain=0.5,
            stability_gain=1.0,
            innovation_gain=0.8,
        ),
        FutureModule(
            name="credibility_filter",
            description="Fake-news and spam inspired trust scoring for web inputs.",
            weight=0.95,
            heat_cost=0.5,
            energy_gain=0.4,
            stability_gain=1.3,
            innovation_gain=0.7,
        ),
        FutureModule(
            name="memory_recommender",
            description="Recommendation-style next-action and memory ranking module.",
            weight=1.05,
            heat_cost=0.6,
            energy_gain=0.7,
            stability_gain=1.1,
            innovation_gain=0.9,
        ),
    ]


def simulate_future_system(config: AgentConfig, steps: int = 12, modules: list[FutureModule] | None = None) -> SimulationResult:
    modules = modules or build_future_modules()
    perf = PerformanceState(
        writing_speed=max(0.6, config.writing_speed),
        activity_level=max(0.7, config.activity_level),
        energy=max(10.0, config.base_energy),
        heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
        burn_rate=max(1.0, config.burn_rate),
        heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
    )

    temp_sum = 0.0
    stability = 0.0
    speed = 0.0
    innovation = 0.0
    module_scores = {m.name: 0.0 for m in modules}

    for turn in range(1, max(1, steps) + 1):
        turn_temp = perf.tune_for_turn(config.temperature, turn, steps)
        temp_sum += turn_temp
        turn_speed = perf.writing_speed * perf.activity_level * (1.0 + max(0.0, perf.energy - 40.0) / 220.0)
        speed += turn_speed

        for mod in modules:
            heat_penalty = 1.0 / (1.0 + (perf.heat / 90.0))
            score = mod.weight * heat_penalty
            module_scores[mod.name] += score
            innovation += mod.innovation_gain * score
            stability += mod.stability_gain * score
            perf.heat += mod.heat_cost * score
            perf.energy = min(300.0, perf.energy + (mod.energy_gain * score))

        perf.passive_cooling()

    norm = float(max(1, steps))
    return SimulationResult(
        steps=max(1, steps),
        final_energy=perf.energy,
        final_heat=perf.heat,
        avg_temperature=temp_sum / norm,
        stability_index=stability / norm,
        speed_index=speed / norm,
        innovation_index=innovation / norm,
        module_scores={k: round(v, 3) for k, v in module_scores.items()},
    )


# ============================================================
# Memory
# ============================================================

class MemoryStore:
    def __init__(self, db_path: Path, max_rows: int = 5000) -> None:
        self.db_path = db_path
        self.max_rows = max(100, max_rows)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                last_accessed TEXT DEFAULT '',
                access_count INTEGER DEFAULT 0
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory(tags)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at)")

        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(key, value, tags, content='memory', content_rowid='id')
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts(rowid, key, value, tags)
                    VALUES (new.id, new.key, new.value, new.tags);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, key, value, tags)
                    VALUES ('delete', old.id, old.key, old.value, old.tags);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, key, value, tags)
                    VALUES ('delete', old.id, old.key, old.value, old.tags);
                    INSERT INTO memory_fts(rowid, key, value, tags)
                    VALUES (new.id, new.key, new.value, new.tags);
                END
                """
            )
        except sqlite3.Error:
            pass
        self.conn.commit()

    def add(self, item: MemoryItem) -> None:
        accessed = item.last_accessed or item.created_at
        self.conn.execute(
            """
            INSERT INTO memory (key, value, tags, created_at, importance, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.key,
                item.value,
                item.tags,
                item.created_at,
                max(0.0, min(1.0, item.importance)),
                accessed,
                1,
            ),
        )
        self.conn.commit()
        self.prune_memory()

    def prune_memory(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM memory").fetchone()
        count = int(row["n"])
        if count <= self.max_rows:
            return 0
        remove_n = count - self.max_rows
        self.conn.execute(
            """
            DELETE FROM memory
            WHERE id IN (
                SELECT id FROM memory
                ORDER BY importance ASC, created_at ASC
                LIMIT ?
            )
            """,
            (remove_n,),
        )
        self.conn.commit()
        return remove_n

    def latest(self, limit: int = 12) -> list[MemoryItem]:
        rows = self.conn.execute(
            """
            SELECT key, value, tags, created_at, importance, COALESCE(last_accessed, created_at) AS last_accessed
            FROM memory
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    def query(self, text: str, limit: int = 8) -> list[MemoryItem]:
        text = text.strip()
        if not text:
            return self.latest(limit=limit)
        rows: list[sqlite3.Row] = []
        try:
            q = self._fts_query(text)
            rows = self.conn.execute(
                """
                SELECT m.id, m.key, m.value, m.tags, m.created_at, m.importance,
                       COALESCE(m.last_accessed, m.created_at) AS last_accessed
                FROM memory_fts f
                JOIN memory m ON m.id = f.rowid
                WHERE memory_fts MATCH ?
                ORDER BY m.importance DESC, m.id DESC
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
        except sqlite3.Error:
            rows = []

        if not rows:
            pattern = f"%{text}%"
            rows = self.conn.execute(
                """
                SELECT id, key, value, tags, created_at, importance,
                       COALESCE(last_accessed, created_at) AS last_accessed
                FROM memory
                WHERE key LIKE ? OR value LIKE ? OR tags LIKE ?
                ORDER BY importance DESC, id DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit),
            ).fetchall()

        self._touch([int(row["id"]) for row in rows if "id" in row.keys()])
        return [MemoryItem(**{k: row[k] for k in row.keys() if k != "id"}) for row in rows]

    def reminisce(self, goal: str, limit: int = 8) -> list[MemoryItem]:
        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_:-]+", goal) if t}
        rows = self.conn.execute(
            """
            SELECT id, key, value, tags, created_at, importance,
                   COALESCE(last_accessed, created_at) AS last_accessed,
                   COALESCE(access_count, 0) AS access_count
            FROM memory
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()

        now_ts = time.time()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            content = f"{row['key']} {row['value']} {row['tags']}".lower()
            overlap = sum(1 for t in tokens if t in content)
            relevance = min(1.0, overlap / max(1, len(tokens))) if tokens else 0.0
            recency = 0.2
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                age_hours = max(0.0, (now_ts - created.timestamp()) / 3600.0)
                recency = 1.0 / (1.0 + age_hours / 24.0)
            except Exception:
                pass
            importance = float(row["importance"] or 0.5)
            usage = min(1.0, float(row["access_count"] or 0) / 12.0)
            score = (0.45 * relevance) + (0.25 * recency) + (0.2 * importance) + (0.1 * usage)
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_rows = [row for _, row in scored[:limit]]
        self._touch([int(row["id"]) for row in top_rows])
        return [
            MemoryItem(
                key=row["key"],
                value=row["value"],
                tags=row["tags"],
                created_at=row["created_at"],
                importance=float(row["importance"] or 0.5),
                last_accessed=row["last_accessed"],
            )
            for row in top_rows
        ]

    def remembrance_guide(self, goal: str, limit: int = 6) -> str:
        memories = self.reminisce(goal, limit=limit)
        if not memories:
            return "No remembrance guide yet."
        lines = []
        for item in memories:
            lines.append(
                f"- Use {item.key} ({item.tags}) because it matched prior success. "
                f"Importance={item.importance:.2f}. Note: {truncate(item.value, 180)}"
            )
        return "\n".join(lines)

    def soften_negative_memories(self, forgetting_rate: float = 0.2, limit: int = 24) -> int:
        rate = max(0.01, min(0.95, forgetting_rate))
        rows = self.conn.execute(
            """
            SELECT id, value, importance
            FROM memory
            WHERE lower(tags) LIKE '%negative%'
               OR lower(tags) LIKE '%impact%'
               OR lower(tags) LIKE '%pain%'
               OR lower(key) LIKE '%incident%'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        softened = 0
        for row in rows:
            new_importance = max(0.03, float(row["importance"] or 0.5) * (1.0 - rate))
            value = str(row["value"] or "")
            if not value.startswith("[attenuated] "):
                value = "[attenuated] " + truncate(value, 480)
            self.conn.execute(
                "UPDATE memory SET value = ?, importance = ?, last_accessed = ? WHERE id = ?",
                (value, new_importance, utc_now(), row["id"]),
            )
            softened += 1
        if softened:
            self.conn.commit()
        return softened

    def _fts_query(self, text: str) -> str:
        terms = [re.sub(r"[^A-Za-z0-9_:-]+", "", t) for t in text.split()]
        terms = [t for t in terms if t]
        return " OR ".join(terms[:8]) or "memory"

    def _touch(self, ids: list[int]) -> None:
        if not ids:
            return
        now = utc_now()
        placeholders = ",".join("?" for _ in ids)
        params = [now, *ids]
        self.conn.execute(
            f"UPDATE memory SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1 WHERE id IN ({placeholders})",
            params,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ============================================================
# Task graph
# ============================================================

@dataclass(slots=True)
class TaskNode:
    name: str
    description: str
    deps: set[str] = field(default_factory=set)
    state: str = "pending"
    notes: str = ""


class TaskGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}

    def add(self, name: str, description: str, deps: list[str] | None = None) -> None:
        self.nodes[name] = TaskNode(name=name, description=description, deps=set(deps or []))

    def mark_done(self, name: str, notes: str = "") -> None:
        if name in self.nodes:
            self.nodes[name].state = "done"
            if notes:
                self.nodes[name].notes = notes

    def next_ready(self) -> list[TaskNode]:
        ready: list[TaskNode] = []
        for node in self.nodes.values():
            if node.state in {"done", "running", "blocked"}:
                continue
            if all(self.nodes.get(dep) and self.nodes[dep].state == "done" for dep in node.deps):
                node.state = "ready"
                ready.append(node)
        return ready

    def summary(self) -> str:
        if not self.nodes:
            return "No task graph."
        lines = []
        for node in self.nodes.values():
            dep_str = ", ".join(sorted(node.deps)) if node.deps else "-"
            lines.append(f"- {node.name} [{node.state}] deps={dep_str} :: {node.description}")
        return "\n".join(lines)


# ============================================================
# Tools
# ============================================================

@dataclass(slots=True)
class ToolContext:
    workspace: Path
    max_output_chars: int = 16000


class ToolError(RuntimeError):
    pass


ToolFn = Callable[[ToolContext, dict[str, Any]], str]


def _resolve_path(ctx: ToolContext, path: str) -> Path:
    candidate = (ctx.workspace / path).resolve()
    workspace = ctx.workspace.resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ToolError(f"Path escapes workspace: {path}")
    return candidate


def _run_subprocess(command: list[str], cwd: Path, timeout: int) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return json_dumps(
        {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    )


def _safe_regex_sub(pattern: str, repl: str, text: str, count: int = 0) -> tuple[str, int]:
    compiled = re.compile(pattern, flags=re.MULTILINE | re.DOTALL)
    return compiled.subn(repl, text, count=count)


def _fetch_search_html(query: str, timeout: int = 20) -> str:
    endpoint = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    return fetch_text_url(endpoint, timeout=timeout, max_chars=250000)


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, ToolFn] = {}
        self.register_defaults()

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        output = self._tools[name](self.context, arguments)
        return truncate(output, self.context.max_output_chars)

    def schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "list_files",
                "description": "List files and directories under a path in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "write_file",
                "description": "Write text content to a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "search_code",
                "description": "Regex search through text files in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "patch_regex",
                "description": "Apply a regex replacement to a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                        "replacement": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["path", "pattern", "replacement"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_python",
                "description": "Run inline Python code in the workspace with timeout.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_shell",
                "description": "Run a safe allowlisted shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "lint_python",
                "description": "Compile a Python file and return syntax errors if present.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_tests",
                "description": "Run pytest if available, or python -m unittest discover as fallback.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "timeout": {"type": "integer"}},
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web for current information and implementation ideas.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "fetch_url",
                "description": "Fetch visible text from a public URL.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "security_scan",
                "description": "Scan code text for risky patterns such as eval/exec or shell hazards.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def register_defaults(self) -> None:
        self.register("list_files", tool_list_files)
        self.register("read_file", tool_read_file)
        self.register("write_file", tool_write_file)
        self.register("search_code", tool_search_code)
        self.register("patch_regex", tool_patch_regex)
        self.register("run_python", tool_run_python)
        self.register("run_shell", tool_run_shell)
        self.register("lint_python", tool_lint_python)
        self.register("run_tests", tool_run_tests)
        self.register("web_search", tool_web_search)
        self.register("fetch_url", tool_fetch_url)
        self.register("security_scan", tool_security_scan)


def tool_list_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    target = _resolve_path(ctx, args.get("path", "."))
    if not target.exists():
        return f"Path not found: {target}"
    items = []
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        kind = "FILE" if item.is_file() else "DIR"
        rel = item.relative_to(ctx.workspace)
        items.append(f"{kind}\t{rel}")
    return "\n".join(items) if items else "(empty)"


def tool_read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = _resolve_path(ctx, args["path"])
    max_chars = int(args.get("max_chars", 12000))
    if not path.exists() or not path.is_file():
        return f"File not found: {path}"
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def tool_write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = _resolve_path(ctx, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.get("append", False) else "w"
    content = args["content"]
    with path.open(mode, encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path.relative_to(ctx.workspace)}"


def tool_search_code(ctx: ToolContext, args: dict[str, Any]) -> str:
    try:
        pattern = re.compile(args["pattern"])
    except re.error as exc:
        return f"Regex error: {exc}"
    root = _resolve_path(ctx, args.get("path", "."))
    hits: list[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".sqlite3", ".db", ".exe", ".dll"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = file_path.relative_to(ctx.workspace)
                hits.append(f"{rel}:{i}: {line.strip()}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n...[limited]"
    return "\n".join(hits) if hits else "No matches"


def tool_patch_regex(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = _resolve_path(ctx, args["path"])
    if not path.exists() or not path.is_file():
        return f"File not found: {path}"
    old = path.read_text(encoding="utf-8", errors="replace")
    new, count = _safe_regex_sub(
        pattern=args["pattern"],
        repl=args["replacement"],
        text=old,
        count=int(args.get("count", 0)),
    )
    if count == 0:
        return f"No changes made to {path.relative_to(ctx.workspace)}"
    path.write_text(new, encoding="utf-8")
    return f"Patched {count} occurrence(s) in {path.relative_to(ctx.workspace)}"


def tool_run_python(ctx: ToolContext, args: dict[str, Any]) -> str:
    timeout = int(args.get("timeout", 20))
    return _run_subprocess([sys.executable, "-c", args["code"]], cwd=ctx.workspace, timeout=timeout)


def tool_run_shell(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = args["command"]
    timeout = int(args.get("timeout", 20))
    if any(token in command for token in ["&&", "||", "|", ";", ">", "<"]):
        raise ToolError("Shell control operators are blocked")
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ToolError("Empty command")
    allowed_roots = {"python", "py", "pytest", "pip", "node", "npm", "git"}
    root = parts[0].lower()
    if root not in allowed_roots:
        raise ToolError(f"Command not allowed: {parts[0]}")
    if root == "git" and len(parts) > 1 and parts[1] not in {"status", "diff", "log", "branch"}:
        raise ToolError("Only read-only git commands are allowed")
    return _run_subprocess(parts, cwd=ctx.workspace, timeout=timeout)


def tool_lint_python(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = _resolve_path(ctx, args["path"])
    if not path.exists() or not path.is_file():
        return f"File not found: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(text, str(path), "exec")
        return json_dumps({"path": str(path.relative_to(ctx.workspace)), "ok": True})
    except SyntaxError as exc:
        return json_dumps(
            {
                "path": str(path.relative_to(ctx.workspace)),
                "ok": False,
                "line": exc.lineno,
                "offset": exc.offset,
                "msg": exc.msg,
                "text": exc.text,
            }
        )


def tool_run_tests(ctx: ToolContext, args: dict[str, Any]) -> str:
    timeout = int(args.get("timeout", 60))
    target = args.get("path", ".")
    path = _resolve_path(ctx, target)
    if (path / "pytest.ini").exists() or any(path.glob("test_*.py")) or any((path / "tests").glob("**/*.py")):
        try:
            return _run_subprocess(["pytest", "-q"], cwd=path, timeout=timeout)
        except FileNotFoundError:
            pass
    return _run_subprocess([sys.executable, "-m", "unittest", "discover"], cwd=path, timeout=timeout)


def tool_web_search(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = args["query"]
    limit = max(1, min(10, int(args.get("limit", 5))))
    try:
        html = _fetch_search_html(query)
    except Exception as exc:
        return json_dumps({"query": query, "error": f"{type(exc).__name__}: {exc}", "results": []})

    pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.I)
    hits = []
    for m in pattern.finditer(html):
        title = re.sub("<.*?>", "", m.group("title"))
        href = m.group("href")
        hits.append({"title": title, "url": href})
        if len(hits) >= limit:
            break
    return json_dumps({"query": query, "results": hits})


def tool_fetch_url(ctx: ToolContext, args: dict[str, Any]) -> str:
    url = args["url"]
    max_chars = int(args.get("max_chars", 8000))
    try:
        body = fetch_text_url(url, timeout=20, max_chars=max_chars * 6)
    except Exception as exc:
        return json_dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    text = re.sub(r"<script.*?</script>", "", body, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def tool_security_scan(ctx: ToolContext, args: dict[str, Any]) -> str:
    text = args.get("text")
    if not text and args.get("path"):
        path = _resolve_path(ctx, args["path"])
        if not path.exists() or not path.is_file():
            return f"File not found: {path}"
        text = path.read_text(encoding="utf-8", errors="replace")
    text = text or ""
    checks = {
        "eval": r"\beval\s*\(",
        "exec": r"\bexec\s*\(",
        "pickle_loads": r"pickle\.(load|loads)\s*\(",
        "os_system": r"os\.system\s*\(",
        "subprocess_shell_true": r"subprocess\.[A-Za-z_]+\([^\)]*shell\s*=\s*True",
        "hardcoded_api_key": r"(sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY\s*=\s*['\"][^'\"]+)",
    }
    findings = []
    for name, pattern in checks.items():
        if re.search(pattern, text, flags=re.I | re.S):
            findings.append(name)
    return json_dumps({"findings": findings, "safe": not findings})


# ============================================================
# OpenAI client
# ============================================================

class OpenAIClient:
    def __init__(self, config: AgentConfig) -> None:
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.config = config
        self.api_key = config.api_key
        self.api_base = config.api_base.rstrip("/")

    def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.retry_attempts:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
        raise RuntimeError(f"API request failed after retries: {last_error}") from last_error

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
        function_outputs: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelTurn:
        if self.config.use_responses_api:
            return self._complete_responses(messages, tools, previous_response_id, function_outputs, temperature)
        return self._complete_chat(messages, tools, temperature)

    def _complete_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
        function_outputs: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "tools": tools,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if previous_response_id and function_outputs:
            payload["previous_response_id"] = previous_response_id
            payload["input"] = function_outputs
        else:
            payload["input"] = messages
        raw = self._request_json("/responses", payload)
        return self._parse_responses_turn(raw)

    def _parse_responses_turn(self, raw: dict[str, Any]) -> ModelTurn:
        output_items = raw.get("output", []) or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        output_text = raw.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            text_parts.append(output_text)
        for item in output_items:
            item_type = item.get("type")
            if item_type == "function_call":
                args_text = item.get("arguments") or "{}"
                try:
                    args = json.loads(args_text) if isinstance(args_text, str) else args_text
                except json.JSONDecodeError:
                    args = {"_raw": args_text}
                tool_calls.append(
                    ToolCall(
                        id=item.get("id") or "",
                        call_id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name") or "",
                        arguments=args if isinstance(args, dict) else {"value": args},
                    )
                )
                continue
            if item_type == "message":
                for content_item in item.get("content", []) or []:
                    if content_item.get("type") in {"output_text", "text"}:
                        text = content_item.get("text", "")
                        if text:
                            text_parts.append(text)
        return ModelTurn(content="\n".join(part for part in text_parts if part).strip(), tool_calls=tool_calls, raw=raw)

    def _complete_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> ModelTurn:
        chat_tools = []
        for t in tools:
            if t.get("type") == "function":
                fn = dict(t)
                fn.pop("type", None)
                chat_tools.append({"type": "function", "function": fn})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": chat_tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        raw = self._request_json("/chat/completions", payload)
        message = raw["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {})
            arguments = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                parsed_args = {"_raw": arguments}
            tool_calls.append(
                ToolCall(
                    id=call.get("id", ""),
                    call_id=call.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=parsed_args if isinstance(parsed_args, dict) else {"value": parsed_args},
                )
            )
        return ModelTurn(content=content, tool_calls=tool_calls, raw=raw)


# ============================================================
# Agent
# ============================================================

SYSTEM_PROMPT = """You are Codex Unified, an advanced autonomous engineering system.

Mode:
1) Planner: produce a concise, grounded plan.
2) Builder: use tools, inspect files, edit carefully, and validate.
3) Reviewer: critique code quality, correctness, resilience, and security. Then patch.

Rules:
- Prefer evidence over vibes.
- Use tools when facts are missing.
- Prefer internet research for current implementation ideas when relevant.
- Avoid repeating the same failed tool call more than twice.
- When editing code, validate it.
- Keep changes concrete and testable.
- End with a final answer only when you have enough evidence.
"""

GOOGLE_INSPIRED_IDEAS = [
    "Iterative planning with fast feedback loops.",
    "Retrieval-first memory grounding before major decisions.",
    "Self-critique checkpoints to reduce drift and hallucination.",
    "Tool orchestration from explicit objectives.",
]


class CodexUnifiedAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.memory = MemoryStore(config.memory_db_path, max_rows=config.memory_max_rows)
        self.tools = ToolRegistry(ToolContext(config.workspace, config.max_tool_output_chars))
        self.client = OpenAIClient(config)
        self.task_graph = TaskGraph()
        self.performance = PerformanceState(
            writing_speed=max(0.6, config.writing_speed),
            activity_level=max(0.7, config.activity_level),
            energy=max(10.0, config.base_energy),
            heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
            burn_rate=max(1.0, config.burn_rate),
            heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
        )
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = threading.Event()

    def close(self) -> None:
        self.stop_background_learning()
        self.memory.close()

    def _log(self, *parts: Any) -> None:
        if self.config.verbose:
            print("[codex]", *parts, file=sys.stderr)

    def _memory_context(self, goal: str) -> str:
        recent = self.memory.latest(limit=6)
        related = self.memory.query(goal, limit=6)
        remembered = self.memory.reminisce(goal, limit=self.config.reminiscence_depth)
        merged: list[MemoryItem] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in recent + related + remembered:
            sig = (item.key, item.value, item.tags, item.created_at)
            if sig not in seen:
                merged.append(item)
                seen.add(sig)
        if not merged:
            return "No memory yet."
        lines = [f"- [{m.created_at}] {m.key} ({m.tags}) imp={m.importance:.2f}: {truncate(m.value, 180)}" for m in merged]
        return "\n".join(lines)

    def _innovation_guide(self, goal: str) -> str:
        remembrance = self.memory.remembrance_guide(goal, limit=6)
        ideas = "\n".join(f"- {idea}" for idea in GOOGLE_INSPIRED_IDEAS)
        gfg = "\n".join(
            f"- {item.title}: {item.benefit} Hint: {item.implementation_hint}"
            for item in select_compatible_geeksforgeeks_modules(limit=6)
        )
        return (
            f"Google-inspired strategy:\n{ideas}\n\n"
            f"Compatible GeeksforGeeks ideas:\n{gfg}\n\n"
            f"Remembrance guide:\n{remembrance}"
        )

    def _seed_task_graph(self, goal: str) -> None:
        self.task_graph.add("inspect", "Inspect the workspace and understand code structure.")
        self.task_graph.add("plan", f"Plan execution for goal: {goal}", deps=["inspect"])
        self.task_graph.add("implement", "Apply code changes and improvements.", deps=["plan"])
        self.task_graph.add("validate", "Run syntax checks and tests when possible.", deps=["implement"])
        self.task_graph.add("review", "Critique and patch remaining issues.", deps=["validate"])

    def start_background_learning(self) -> None:
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._bg_stop.clear()

        def _worker() -> None:
            topics = [
                "agent memory ranking ideas",
                "prompt injection defense patterns",
                "text summarization techniques",
                "lightweight recommendation systems",
            ]
            idx = 0
            while not self._bg_stop.is_set():
                topic = topics[idx % len(topics)]
                idx += 1
                try:
                    result = tool_web_search(self.tools.context, {"query": topic, "limit": 3})
                    self.memory.add(MemoryItem(key="background_learning", value=result, tags="background,web", importance=0.35))
                except Exception:
                    pass
                self._bg_stop.wait(600)

        self._bg_thread = threading.Thread(target=_worker, name="codex-bg-learning", daemon=True)
        self._bg_thread.start()

    def stop_background_learning(self) -> None:
        self._bg_stop.set()
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=0.2)
        self._bg_thread = None

    def run(self, goal: str) -> AgentRunResult:
        self._seed_task_graph(goal)
        mem_ctx = self._memory_context(goal)
        innovation_guide = self._innovation_guide(goal)
        ready_now = self.task_graph.next_ready()
        ready_text = ", ".join(node.name for node in ready_now) if ready_now else "none"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Workspace: {self.config.workspace}\n"
                    f"Memory context:\n{mem_ctx}\n\n"
                    f"Innovation guide:\n{innovation_guide}\n\n"
                    f"Task graph:\n{self.task_graph.summary()}\n\n"
                    f"Ready tasks now: {ready_text}\n"
                    f"Performance state: {self.performance.snapshot()}\n"
                    "Start by inspecting the workspace, then make a concise plan, then execute it with evidence."
                ),
            },
        ]

        final_answer = ""
        total_tool_calls = 0
        repeated_calls = Counter()
        tool_failures = Counter()
        previous_response_id: str | None = None
        function_outputs: list[dict[str, Any]] | None = None

        for turn in range(1, self.config.max_turns + 1):
            turn_temp = self.performance.tune_for_turn(self.config.temperature, turn, self.config.max_turns)
            self._log(
                f"turn={turn} previous_response_id={previous_response_id!r} "
                f"temperature={turn_temp:.2f} perf={self.performance.snapshot()}"
            )

            if turn > 1 and turn % 2 == 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Remembrance refresh: keep concise writing, grounded tool use, and validate edits. "
                            f"Current performance: {self.performance.snapshot()}"
                        ),
                    }
                )

            result = self.client.complete(
                messages=messages,
                tools=self.tools.schema(),
                previous_response_id=previous_response_id,
                function_outputs=function_outputs,
                temperature=turn_temp,
            )
            previous_response_id = result.raw.get("id") or previous_response_id
            function_outputs = None

            if result.content:
                final_answer = result.content

            if not result.tool_calls:
                if turn >= 2:
                    break
                messages.append({"role": "user", "content": "No tool calls detected. Continue with concrete execution or finalize if done."})
                continue

            outputs: list[dict[str, Any]] = []
            for call in result.tool_calls:
                total_tool_calls += 1
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                repeated_calls[key] += 1
                if repeated_calls[key] > 2:
                    tool_out = f"Tool skipped ({call.name}): repeated identical call too many times."
                else:
                    try:
                        tool_out = self.tools.call(call.name, call.arguments)
                    except Exception as exc:
                        tool_failures[call.name] += 1
                        tool_out = f"Tool error ({call.name}): {exc}\n{traceback.format_exc(limit=2)}"
                self._log(f"tool={call.name} repeated={repeated_calls[key]} failures={tool_failures[call.name]}")
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": tool_out})
                messages.append({"role": "tool", "tool_call_id": call.id or call.call_id, "content": tool_out})

            function_outputs = outputs

            if turn == 1:
                self.task_graph.mark_done("inspect", "Workspace inspection performed.")
                self.task_graph.mark_done("plan", "Initial plan produced.")
            elif turn == 2:
                self.task_graph.mark_done("implement", "Implementation phase active.")
            elif turn >= 3:
                self.task_graph.mark_done("validate", "Validation phase reached.")

            if turn % self.config.reviewer_every == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reviewer phase: identify flaws, risky assumptions, syntax/runtime issues, "
                            "and missing validations. Patch them before finalizing."
                        ),
                    }
                )

        self.task_graph.mark_done("review", "Review completed or loop ended.")
        self.memory.soften_negative_memories(self.config.negative_forgetting_rate)
        self.memory.add(MemoryItem(key="goal", value=goal, tags="run", importance=0.85))
        self.memory.add(MemoryItem(key="result", value=truncate(final_answer or "No final answer produced.", 1200), tags="run", importance=0.9))
        self.memory.add(MemoryItem(key="guide", value=truncate(innovation_guide, 1200), tags="guide,gfg", importance=0.7))

        return AgentRunResult(
            final_answer=final_answer or "No final answer produced.",
            turns_used=turn,
            tool_calls=total_tool_calls,
            task_summary=self.task_graph.summary(),
        )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex Unified - advanced AI coding agent")
    parser.add_argument("--goal", default="", help="Primary goal for the agent")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Workspace root path")
    parser.add_argument("--max-turns", type=int, default=None, help="Override maximum reasoning turns")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--activity-level", type=float, default=None, help="Override activity level")
    parser.add_argument("--writing-speed", type=float, default=None, help="Override writing speed")
    parser.add_argument("--base-energy", type=float, default=None, help="Override base energy")
    parser.add_argument("--heat-resistance", type=float, default=None, help="Override heat resistance")
    parser.add_argument("--memory-max-rows", type=int, default=None, help="Override memory pruning ceiling")
    parser.add_argument("--resilience-factor", type=float, default=None, help="Override resilience factor")
    parser.add_argument("--negative-forgetting-rate", type=float, default=None, help="Override forgetting rate")
    parser.add_argument("--background-learning", action="store_true", help="Start background web-learning loop during agent run")
    parser.add_argument("--simulate", action="store_true", help="Run future-module simulation and exit")
    parser.add_argument("--simulation-steps", type=int, default=12, help="Simulation step count")
    parser.add_argument("--list-geeksforgeeks-modules", action="store_true", help="List compatible GeeksforGeeks-inspired modules and exit")
    parser.add_argument("--chat-completions", action="store_true", help="Use legacy chat/completions fallback")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON result")
    parser.add_argument("--verbose", action="store_true", help="Verbose stderr logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig.from_env()
    if args.model:
        config.model = args.model
    if args.workspace:
        config.workspace = Path(args.workspace).resolve()
        config.workspace.mkdir(parents=True, exist_ok=True)
    if args.max_turns is not None:
        config.max_turns = args.max_turns
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.activity_level is not None:
        config.activity_level = args.activity_level
    if args.writing_speed is not None:
        config.writing_speed = args.writing_speed
    if args.base_energy is not None:
        config.base_energy = args.base_energy
    if args.heat_resistance is not None:
        config.heat_resistance = args.heat_resistance
    if args.memory_max_rows is not None:
        config.memory_max_rows = args.memory_max_rows
    if args.resilience_factor is not None:
        config.resilience_factor = args.resilience_factor
    if args.negative_forgetting_rate is not None:
        config.negative_forgetting_rate = args.negative_forgetting_rate
    if args.chat_completions:
        config.use_responses_api = False
    if args.verbose:
        config.verbose = True

    if args.simulate:
        sim = simulate_future_system(config, steps=max(1, int(args.simulation_steps)))
        print(
            json_dumps(
                {
                    "version": VERSION,
                    "mode": "simulation",
                    "steps": sim.steps,
                    "final_energy": round(sim.final_energy, 3),
                    "final_heat": round(sim.final_heat, 3),
                    "avg_temperature": round(sim.avg_temperature, 4),
                    "stability_index": round(sim.stability_index, 4),
                    "speed_index": round(sim.speed_index, 4),
                    "innovation_index": round(sim.innovation_index, 4),
                    "module_scores": sim.module_scores,
                }
            )
        )
        return 0

    if args.list_geeksforgeeks_modules:
        items = build_geeksforgeeks_compatible_catalog()
        print(
            json_dumps(
                [
                    {
                        "title": item.title,
                        "category": item.category,
                        "benefit": item.benefit,
                        "implementation_hint": item.implementation_hint,
                        "source_url": item.source_url,
                    }
                    for item in items
                ]
            )
        )
        return 0

    if not args.goal:
        print("Error: --goal is required unless using a listing or simulation mode.", file=sys.stderr)
        return 2

    agent = CodexUnifiedAgent(config)
    try:
        if args.background_learning:
            agent.start_background_learning()
        result = agent.run(args.goal)
    finally:
        agent.close()

    if args.json:
        print(
            json_dumps(
                {
                    "version": VERSION,
                    "model": config.model,
                    "workspace": str(config.workspace),
                    "turns_used": result.turns_used,
                    "tool_calls": result.tool_calls,
                    "task_summary": result.task_summary,
                    "final_answer": result.final_answer,
                }
            )
        )
    else:
        print(f"[{VERSION}] model={config.model} turns={result.turns_used} tool_calls={result.tool_calls}")
        print()
        print("Task summary:")
        print(result.task_summary)
        print()
        print("Final answer:")
        print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
