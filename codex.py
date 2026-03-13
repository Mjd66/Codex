#!/usr/bin/env python3
"""
Codex Unified v3
Single-file autonomous engineering agent with:
- OpenAI Responses API support (with Chat Completions fallback)
- planner / builder / reviewer loop
- sqlite memory with WAL + FTS5 retrieval
- safe workspace tools
- web search + fetch tools
- regex patching, lint/test helpers
- task graph for longer-horizon work
- CLI entrypoint

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


VERSION = "codex_unified_v3"


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


def bytes_to_human(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


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
    resilience_factor: float = 1.0
    negative_forgetting_rate: float = 0.2
    continuity_target: float = 100.0

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
            resilience_factor=float(os.getenv("CODEX_RESILIENCE_FACTOR", "1.0")),
            negative_forgetting_rate=float(os.getenv("CODEX_NEGATIVE_FORGETTING_RATE", "0.2")),
            continuity_target=float(os.getenv("CODEX_CONTINUITY_TARGET", "100")),
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

    def tune_for_turn(self, base_temperature: float, turn: int, max_turns: int) -> float:
        self.burn_to_heat()
        self.absorb_heat_to_energy()

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
class GitHubModuleRef:
    name: str
    repo: str
    url: str
    focus: str
    why_it_matters: str


@dataclass(slots=True)
class OpenAIModuleRef:
    name: str
    url: str
    focus: str
    why_it_matters: str


@dataclass(slots=True)
class VRModuleRef:
    name: str
    url: str
    language_mix: str
    focus: str
    why_it_matters: str


@dataclass(slots=True)
class WebProjectIdeaRef:
    idea_id: int
    level: str
    title: str
    languages: list[str]
    source_url: str


@dataclass(slots=True)
class RequestedAddonRef:
    addon_id: int
    title: str
    primary_python_stack: list[str]
    optional_mixed_languages: list[str]
    focus: str


@dataclass(slots=True)
class NextSentencePredictionResult:
    sentence_a: str
    sentence_b: str
    is_consecutive: bool
    confidence: float
    backend: str
    model: str
    details: dict[str, Any]


@dataclass(slots=True)
class ComplexAbs2MethodRef:
    name: str
    backend_key: str
    preserves_input: bool
    output_dtype: str
    memory_note: str
    source_url: str


@dataclass(slots=True)
class BuiltinComplexModuleRef:
    name: str
    short_description: str
    last_verified: str
    constructor_syntax: list[str]
    arguments: dict[str, str]
    return_value: str
    examples: list[str]
    methods: dict[str, str]
    common_use_cases: list[str]
    real_world_example: list[str]
    related_resources: list[str]
    source_url: str


@dataclass(slots=True)
class EmergingPython2026ModuleRef:
    name: str
    category: str
    summary: str
    python_stack: list[str]
    suggested_application: str
    cited_versions: dict[str, str]
    compliance_notes: list[str]
    source_url: str


@dataclass(slots=True)
class LinkedInPython2025TrendModuleRef:
    trend_id: int
    trend_name: str
    summary: str
    python_stack: list[str]
    suggested_application: str
    learning_path: list[str]
    source_url: str


@dataclass(slots=True)
class VRAutonomyAccessibilityModuleRef:
    module_id: int
    name: str
    category: str
    summary: str
    python_stack: list[str]
    mixed_stack: list[str]
    target_outcome: str
    compliance_notes: list[str]
    sources: list[str]


@dataclass(slots=True)
class VRAutonomySimulationResult:
    steps: int
    automation_reliability: float
    update_safety_score: float
    autopilot_world_stability: float
    haptic_accessibility_score: float
    hearing_accessibility_score: float
    mental_mode_safety_score: float
    overall_safety_score: float


@dataclass(slots=True)
class ImpactResilienceState:
    resilience: float = 1.0
    pain_load: float = 0.0
    impact_load: float = 0.0
    continuity: float = 100.0
    negative_memory_pressure: float = 0.0

    def ingest_event(self, severity: float, emotional_weight: float = 1.0) -> None:
        sev = max(0.0, severity)
        emo = max(0.2, emotional_weight)
        self.pain_load += sev * emo
        self.impact_load += sev
        self.negative_memory_pressure += sev * 0.5
        self.continuity = max(0.0, self.continuity - (sev * max(0.2, 1.1 - self.resilience)))

    def recover(self, perf: PerformanceState, forgetting_rate: float = 0.2) -> None:
        recovery = (0.6 + (perf.energy / 260.0)) * max(0.25, self.resilience)
        self.pain_load = max(0.0, self.pain_load - recovery)
        self.impact_load = max(0.0, self.impact_load - (recovery * 0.8))
        self.negative_memory_pressure = max(0.0, self.negative_memory_pressure - (recovery * forgetting_rate))
        self.continuity = min(100.0, self.continuity + (recovery * 0.55))

    def snapshot(self) -> str:
        return (
            f"resilience={self.resilience:.2f}, pain={self.pain_load:.2f}, impact={self.impact_load:.2f}, "
            f"continuity={self.continuity:.2f}, neg_memory_pressure={self.negative_memory_pressure:.2f}"
        )


@dataclass(slots=True)
class VRSimulationResult:
    steps: int
    foresight_score: float
    avoidance_success_rate: float
    pain_resilience_score: float
    continuity_score: float
    negative_memory_pressure: float
    recommended_stack: list[str]


@dataclass(slots=True)
class VRTimeTravelModuleRef:
    name: str
    python_stack: list[str]
    mixed_stack: list[str]
    focus: str
    source_url: str


@dataclass(slots=True)
class VRTimeTravelSimulationResult:
    steps: int
    timeline_glimpse_score: float
    hazard_avoidance_score: float
    frame_stability_score: float
    latency_risk: float
    recommended_modules: list[str]


def build_vr_module_catalog() -> list[VRModuleRef]:
    # Practical Python + mixed-language stack options for VR foresight and resilience.
    return [
        VRModuleRef(
            name="OpenXR SDK",
            url="https://github.com/KhronosGroup/OpenXR-SDK-Source",
            language_mix="C/C++ core + Python bindings",
            focus="Cross-platform VR runtime integration",
            why_it_matters="Stable sensor/headset input layer for future-outlook modeling.",
        ),
        VRModuleRef(
            name="ONNX Runtime",
            url="https://github.com/microsoft/onnxruntime",
            language_mix="C++ core + Python + C#",
            focus="Fast model inference for prediction in headset loops",
            why_it_matters="Enables low-latency foresight inference in mixed stacks.",
        ),
        VRModuleRef(
            name="Unity ML-Agents",
            url="https://github.com/Unity-Technologies/ml-agents",
            language_mix="C# + Python trainers",
            focus="Scenario generation and avoidance policy training",
            why_it_matters="Good for pre-training avoidant behavior in virtual risk environments.",
        ),
        VRModuleRef(
            name="pybind11",
            url="https://github.com/pybind/pybind11",
            language_mix="C++ + Python",
            focus="High-performance native extensions for critical loops",
            why_it_matters="Lets Python orchestration call optimized C++ VR physics/forecast kernels.",
        ),
        VRModuleRef(
            name="PyO3",
            url="https://github.com/PyO3/pyo3",
            language_mix="Rust + Python",
            focus="Memory-safe high-performance extension modules",
            why_it_matters="Useful for resilient impact-processing modules with safer native memory handling.",
        ),
    ]


def build_semanticproxy_vr_timetravel_modules() -> list[VRTimeTravelModuleRef]:
    src = "https://semanticproxy.com/blog/using-python-for-virtual-reality-development-tips-and-best-practices/"
    return [
        VRTimeTravelModuleRef(
            name="OpenVR Device Bridge",
            python_stack=["Python", "openvr", "NumPy"],
            mixed_stack=["C++ OpenVR SDK", "Python bindings"],
            focus="Headset/controller access for future-state capture and timeline probes",
            source_url=src,
        ),
        VRTimeTravelModuleRef(
            name="PyOpenGL Timeline Renderer",
            python_stack=["Python", "PyOpenGL", "GLFW"],
            mixed_stack=["OpenGL C/C++ shaders", "Python orchestration"],
            focus="Render parallel timeline previews as lightweight VR overlays",
            source_url=src,
        ),
        VRTimeTravelModuleRef(
            name="Pygame Interaction Sandbox",
            python_stack=["Python", "Pygame"],
            mixed_stack=["SDL C backend"],
            focus="Rapid prototyping for interaction rules and time-jump UX",
            source_url=src,
        ),
        VRTimeTravelModuleRef(
            name="Async VR Loop Scheduler",
            python_stack=["Python", "asyncio"],
            mixed_stack=["Native render loops + Python async control"],
            focus="Separate input, prediction, and render tasks to reduce latency",
            source_url=src,
        ),
        VRTimeTravelModuleRef(
            name="Motion Tracking Risk Engine",
            python_stack=["Python", "OpenCV", "TensorFlow"],
            mixed_stack=["ONNX Runtime C++ acceleration"],
            focus="Predict hazardous trajectories and trigger avoidant time-glimpse cues",
            source_url=src,
        ),
        VRTimeTravelModuleRef(
            name="Performance Guardrail Profiler",
            python_stack=["Python", "cProfile", "tracemalloc"],
            mixed_stack=["GPU profiling tools"],
            focus="Track frame rate, latency, and memory to maintain comfort and continuity",
            source_url=src,
        ),
    ]


def build_bestprojectideas_ai50_catalog() -> list[WebProjectIdeaRef]:
    src = "https://bestprojectideas.com/artificial-intelligence-project-ideas/"
    return [
        WebProjectIdeaRef(1, "beginner", "Transfer-Learning Image Classifier", ["Python", "TensorFlow", "PyTorch"], src),
        WebProjectIdeaRef(2, "beginner", "Hybrid FAQ Chat Assistant", ["Python", "JavaScript", "FastAPI"], src),
        WebProjectIdeaRef(3, "beginner", "Spam Message Filter", ["Python", "scikit-learn"], src),
        WebProjectIdeaRef(4, "beginner", "Movie Review Sentiment Scorer", ["Python", "PyTorch", "Hugging Face"], src),
        WebProjectIdeaRef(5, "beginner", "MNIST Digit Recognizer", ["Python", "TensorFlow"], src),
        WebProjectIdeaRef(6, "beginner", "Housing Price Regressor", ["Python", "scikit-learn"], src),
        WebProjectIdeaRef(7, "beginner", "Custom-Handwriting Digit Recognizer", ["Python", "OpenCV", "TensorFlow"], src),
        WebProjectIdeaRef(8, "beginner", "Mini Seq2Seq Translator", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(9, "beginner", "Pretrained Object Detector", ["Python", "PyTorch", "ONNX"], src),
        WebProjectIdeaRef(10, "beginner", "Keyword Voice Command Recognizer", ["Python", "Librosa", "TensorFlow"], src),
        WebProjectIdeaRef(11, "beginner", "Collaborative Movie Recommender", ["Python", "scikit-learn"], src),
        WebProjectIdeaRef(12, "beginner", "Face Identification Verifier", ["Python", "OpenCV"], src),
        WebProjectIdeaRef(13, "beginner", "Fake News Classifier", ["Python", "scikit-learn", "Transformers"], src),
        WebProjectIdeaRef(14, "beginner", "Road Sign Image Classifier", ["Python", "PyTorch", "OpenCV"], src),
        WebProjectIdeaRef(15, "beginner", "Text Emotion Detector", ["Python", "Transformers"], src),
        WebProjectIdeaRef(16, "intermediate", "Domain Transformer Chatbot", ["Python", "PyTorch", "FastAPI"], src),
        WebProjectIdeaRef(17, "intermediate", "Neural Style Transfer Studio", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(18, "intermediate", "Speech-to-Text Transcriber", ["Python", "C++", "ONNX Runtime"], src),
        WebProjectIdeaRef(19, "intermediate", "Automatic Image Caption Generator", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(20, "intermediate", "Video Object Tracker", ["Python", "OpenCV", "C++"], src),
        WebProjectIdeaRef(21, "intermediate", "Neural Melody Generator", ["Python", "MIDI", "PyTorch"], src),
        WebProjectIdeaRef(22, "intermediate", "Deepfake Media Detector", ["Python", "PyTorch", "OpenCV"], src),
        WebProjectIdeaRef(23, "intermediate", "Deep Learning Recommender Engine", ["Python", "TensorFlow"], src),
        WebProjectIdeaRef(24, "intermediate", "Adversarial Attack and Defense Lab", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(25, "intermediate", "Long-Form Text Summarizer", ["Python", "Transformers"], src),
        WebProjectIdeaRef(26, "intermediate", "Voice-Enabled Chatbot", ["Python", "JavaScript", "WebRTC"], src),
        WebProjectIdeaRef(27, "intermediate", "Voice-Controlled Smart Home Simulator", ["Python", "MQTT", "Node.js"], src),
        WebProjectIdeaRef(28, "intermediate", "Medical Image Segmentation Pipeline", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(29, "intermediate", "RL Navigation Simulator", ["Python", "Gymnasium", "Unity C#"], src),
        WebProjectIdeaRef(30, "intermediate", "Real-Time Hand Gesture Recognizer", ["Python", "OpenCV", "MediaPipe"], src),
        WebProjectIdeaRef(31, "intermediate", "Predictive Maintenance Forecaster", ["Python", "scikit-learn", "pandas"], src),
        WebProjectIdeaRef(32, "intermediate", "Generative AI Art Creator", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(33, "intermediate", "Corpus-Based Question Answering", ["Python", "Transformers"], src),
        WebProjectIdeaRef(34, "intermediate", "Time-Series Prediction Workbench", ["Python", "Prophet", "PyTorch"], src),
        WebProjectIdeaRef(35, "intermediate", "AI Data Augmentation Utility", ["Python", "OpenCV", "Transformers"], src),
        WebProjectIdeaRef(36, "advanced", "Multimodal Vision-Language System", ["Python", "PyTorch", "CUDA C++"], src),
        WebProjectIdeaRef(37, "advanced", "From-Scratch Transformer Build", ["Python"], src),
        WebProjectIdeaRef(38, "advanced", "Advanced RL Simulation Agent", ["Python", "C++", "Unity C#"], src),
        WebProjectIdeaRef(39, "advanced", "Neural Architecture Search Prototype", ["Python", "Ray"], src),
        WebProjectIdeaRef(40, "advanced", "Privacy-Preserving Federated Learner", ["Python", "TensorFlow Federated"], src),
        WebProjectIdeaRef(41, "advanced", "Explainable Medical Diagnosis Model", ["Python", "PyTorch", "OpenCV"], src),
        WebProjectIdeaRef(42, "advanced", "Synthetic Data GAN Generator", ["Python", "PyTorch"], src),
        WebProjectIdeaRef(43, "advanced", "Domain LLM Fine-Tuning Project", ["Python", "Transformers", "Rust"], src),
        WebProjectIdeaRef(44, "advanced", "Vision-Based Autonomous Drone Navigation", ["Python", "C++", "AirSim"], src),
        WebProjectIdeaRef(45, "advanced", "Emotion-Aware Virtual Assistant", ["Python", "JavaScript", "C++"], src),
        WebProjectIdeaRef(46, "advanced", "Driving Scene Semantic Segmentation", ["Python", "PyTorch", "CUDA C++"], src),
        WebProjectIdeaRef(47, "advanced", "AI Trading Strategy Backtester", ["Python", "pandas"], src),
        WebProjectIdeaRef(48, "advanced", "Cross-Lingual QA System", ["Python", "Transformers"], src),
        WebProjectIdeaRef(49, "advanced", "Personalized Learning Recommender", ["Python", "TypeScript"], src),
        WebProjectIdeaRef(50, "advanced", "Climate Pattern Analysis and Forecasting", ["Python", "xarray", "PyTorch"], src),
    ]


def build_requested_addons32_catalog() -> list[RequestedAddonRef]:
    return [
        RequestedAddonRef(1, "Chatbots", ["Python", "NLTK", "spaCy", "FastAPI"], ["JavaScript", "TypeScript"], "Rule-based and NLP chatbot systems"),
        RequestedAddonRef(2, "Fake News Detection System", ["Python", "scikit-learn", "TensorFlow", "Transformers"], [], "Detect and classify misinformation in news text"),
        RequestedAddonRef(3, "Stock Market Predictor", ["Python", "pandas", "scikit-learn", "PyTorch"], ["C++"], "Forecast short-horizon stock movement"),
        RequestedAddonRef(4, "Sentiment Predictor", ["Python", "Transformers", "PyTorch"], [], "Predict customer sentiment from text"),
        RequestedAddonRef(5, "Flower Classification Using AI", ["Python", "TensorFlow", "OpenCV"], [], "Classify flowers using CNN-based vision"),
        RequestedAddonRef(6, "Human Activity Recognition System", ["Python", "OpenCV", "NumPy"], ["C++"], "Detect human actions from sensor/video input"),
        RequestedAddonRef(7, "Wine Quality Analyzer", ["Python", "pandas", "scikit-learn"], [], "Predict wine quality from chemistry features"),
        RequestedAddonRef(8, "Object Detector", ["Python", "OpenCV", "PyTorch"], ["C++", "CUDA"], "Detect objects with deep neural networks"),
        RequestedAddonRef(9, "Recommender Engine", ["Python", "pandas", "scikit-learn"], ["TypeScript"], "Personalized recommendation from behavior"),
        RequestedAddonRef(10, "Sales Predictor", ["Python", "pandas", "Prophet", "scikit-learn"], [], "Forecast product sales weekly/monthly"),
        RequestedAddonRef(11, "Image Caption Generator", ["Python", "PyTorch", "Transformers"], [], "Generate captions for input images"),
        RequestedAddonRef(12, "Predicting Fuel Efficiency", ["Python", "TensorFlow", "scikit-learn"], [], "Regress fuel efficiency from vehicle features"),
        RequestedAddonRef(13, "Detecting Spam Emails", ["Python", "scikit-learn", "TensorFlow"], [], "Classify email as spam or legitimate"),
        RequestedAddonRef(14, "Language Translation", ["Python", "Transformers", "sentencepiece"], [], "Machine translation with attention/transformers"),
        RequestedAddonRef(15, "Text Summarization", ["Python", "Transformers"], [], "Extractive/abstractive summarization"),
        RequestedAddonRef(16, "Hate Speech Detection", ["Python", "PyTorch", "Transformers"], [], "Detect offensive and harmful content"),
        RequestedAddonRef(17, "Text Autocorrector", ["Python", "NLTK", "Transformers"], [], "Spelling correction and text repair"),
        RequestedAddonRef(18, "Recognize Car License Plate from a Video", ["Python", "OpenCV", "EasyOCR"], ["C++"], "Realtime plate detection and OCR"),
        RequestedAddonRef(19, "Age Detection", ["Python", "OpenCV", "PyTorch"], [], "Estimate age range from facial image"),
        RequestedAddonRef(20, "Text Generation", ["Python", "PyTorch"], [], "Generate coherent text from prompts"),
        RequestedAddonRef(21, "Lung Cancer Detection", ["Python", "TensorFlow", "OpenCV"], [], "Detect lung anomalies in medical images"),
        RequestedAddonRef(22, "Recipe Recommendation System", ["Python", "pandas", "scikit-learn"], ["JavaScript"], "Suggest recipes from ingredients/preferences"),
        RequestedAddonRef(23, "Personalized Voice Assistant", ["Python", "SpeechRecognition", "pyttsx3"], ["JavaScript"], "Voice-command personal assistant"),
        RequestedAddonRef(24, "Inventory Demand Forecasting", ["Python", "pandas", "scikit-learn", "XGBoost"], [], "Demand prediction for stock planning"),
        RequestedAddonRef(25, "Speech Recognition", ["Python", "Whisper", "Librosa"], ["C++"], "Convert speech to text in realtime"),
        RequestedAddonRef(26, "Credit Card Fraud Detection", ["Python", "scikit-learn", "LightGBM"], [], "Fraud anomaly detection in transactions"),
        RequestedAddonRef(27, "IPL Score Prediction", ["Python", "pandas", "PyTorch"], [], "Predict cricket innings score"),
        RequestedAddonRef(28, "Loan Eligibility Prediction", ["Python", "scikit-learn"], [], "Classify loan approval eligibility"),
        RequestedAddonRef(29, "Reviews Analysis", ["Python", "Transformers", "scikit-learn"], [], "Sentiment and insight mining from reviews"),
        RequestedAddonRef(30, "Sign Language Recognition System", ["Python", "TensorFlow", "OpenCV", "MediaPipe"], [], "Recognize sign gestures to text/speech"),
        RequestedAddonRef(31, "Text Detection and Extraction", ["Python", "OpenCV", "pytesseract"], ["C++"], "OCR pipeline for text in images/documents"),
        RequestedAddonRef(32, "Next Sentence Prediction", ["Python", "Transformers", "BERT"], [], "Predict contextual next sentence"),
    ]


_NSP_RUNTIME_CACHE: dict[str, tuple[Any, Any]] = {}


def build_nsp_bert_module_spec() -> dict[str, Any]:
    return {
        "name": "Next Sentence Prediction using BERT",
        "last_updated": "2025-07-23",
        "summary": (
            "Determine whether sentence B logically follows sentence A. "
            "Useful for QA, summarization, and dialogue systems."
        ),
        "bert_special_tokens": {
            "CLS": "Classification token at the start of the sequence.",
            "SEP": "Separator token between sentence A and sentence B.",
        },
        "approaches": [
            {
                "name": "Sentence Pair Classification",
                "datasets": ["MNLI", "QQP", "QNLI", "SWAG"],
            },
            {
                "name": "Single Sentence Classification",
                "datasets": ["SST-2", "CoLA"],
            },
            {
                "name": "Question Answering",
                "datasets": ["SQuAD 1.1", "SQuAD 2.0"],
            },
        ],
    }


def _nsp_heuristic(sentence_a: str, sentence_b: str) -> NextSentencePredictionResult:
    a_tokens = re.findall(r"[A-Za-z0-9']+", sentence_a.lower())
    b_tokens = re.findall(r"[A-Za-z0-9']+", sentence_b.lower())

    a_set = set(a_tokens)
    b_set = set(b_tokens)
    overlap = (len(a_set & b_set) / max(1, len(a_set | b_set))) if a_set or b_set else 0.0

    discourse_cues = {
        "then", "next", "after", "therefore", "so", "because", "meanwhile", "however", "but", "and"
    }
    cue = 1.0 if any(tok in discourse_cues for tok in b_tokens[:3]) else 0.0

    pronouns = {"he", "she", "they", "it", "him", "her", "them", "his", "their", "its"}
    coref = 1.0 if any(tok in pronouns for tok in b_tokens[:4]) and len(a_tokens) >= 3 else 0.0

    topic_shift = 1.0 if overlap < 0.06 and len(a_tokens) > 4 and len(b_tokens) > 4 else 0.0
    raw_score = (-0.4) + (1.8 * overlap) + (0.45 * cue) + (0.25 * coref) - (0.7 * topic_shift)
    conf = 1.0 / (1.0 + math.exp(-raw_score))

    return NextSentencePredictionResult(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        is_consecutive=conf >= 0.5,
        confidence=round(conf, 6),
        backend="heuristic",
        model="heuristic-nsp-v1",
        details={
            "overlap": round(overlap, 6),
            "discourse_cue": cue,
            "coreference_signal": coref,
            "topic_shift_penalty": topic_shift,
            "raw_score": round(raw_score, 6),
        },
    )


def run_nsp_prediction(
    sentence_a: str,
    sentence_b: str,
    model_name: str = "bert-base-uncased",
    backend: str = "auto",
) -> NextSentencePredictionResult:
    backend = backend.strip().lower()
    if backend not in {"auto", "transformers", "heuristic"}:
        raise ValueError("backend must be one of: auto, transformers, heuristic")

    if backend in {"auto", "transformers"}:
        try:
            import importlib

            torch_mod = importlib.import_module("torch")
            transformers_mod = importlib.import_module("transformers")
            auto_tokenizer = getattr(transformers_mod, "AutoTokenizer")
            auto_model = getattr(transformers_mod, "AutoModelForNextSentencePrediction")

            if model_name in _NSP_RUNTIME_CACHE:
                tokenizer, model = _NSP_RUNTIME_CACHE[model_name]
            else:
                tokenizer = auto_tokenizer.from_pretrained(model_name)
                model = auto_model.from_pretrained(model_name)
                model.eval()
                _NSP_RUNTIME_CACHE[model_name] = (tokenizer, model)

            encoded = tokenizer(
                sentence_a,
                sentence_b,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch_mod.no_grad():
                logits = model(**encoded).logits[0]
                probs = torch_mod.softmax(logits, dim=-1)
            conf = float(probs[1].item())

            return NextSentencePredictionResult(
                sentence_a=sentence_a,
                sentence_b=sentence_b,
                is_consecutive=conf >= 0.5,
                confidence=round(conf, 6),
                backend="transformers",
                model=model_name,
                details={
                    "label_definition": {"0": "not_next", "1": "is_next"},
                    "note": "Confidence is P(is_next) from BERT NSP head.",
                },
            )
        except Exception as exc:  # noqa: BLE001
            heuristic = _nsp_heuristic(sentence_a, sentence_b)
            heuristic.backend = "heuristic_fallback"
            heuristic.details["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            heuristic.details["requested_model"] = model_name
            if backend == "transformers":
                heuristic.details["forced_backend"] = True
            return heuristic

    return _nsp_heuristic(sentence_a, sentence_b)


def build_complex_abs2_module_catalog() -> list[ComplexAbs2MethodRef]:
    src = "https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray"
    return [
        ComplexAbs2MethodRef(
            name="NumPy abs + in-place square",
            backend_key="numpy_abs_square",
            preserves_input=True,
            output_dtype="real",
            memory_note="Allocates one real output array; squares in place to avoid extra temp.",
            source_url=src,
        ),
        ComplexAbs2MethodRef(
            name="In-place complex overwrite",
            backend_key="numpy_inplace_complex",
            preserves_input=False,
            output_dtype="real-view",
            memory_note="Overwrites input complex buffer for lowest overhead when mutation is allowed.",
            source_url=src,
        ),
        ComplexAbs2MethodRef(
            name="Numba vectorized ufunc",
            backend_key="numba_vectorize",
            preserves_input=True,
            output_dtype="real",
            memory_note="Computes abs2 directly per element with no large intermediate arrays.",
            source_url=src,
        ),
        ComplexAbs2MethodRef(
            name="Conjugate multiply",
            backend_key="conjugate_multiply",
            preserves_input=True,
            output_dtype="real",
            memory_note="Simple expression but keeps complex intermediate buffer.",
            source_url=src,
        ),
        ComplexAbs2MethodRef(
            name="Float-view dot",
            backend_key="float_view_dot",
            preserves_input=True,
            output_dtype="real",
            memory_note="Uses float view and einsum; may require contiguous conversion copy.",
            source_url=src,
        ),
    ]


_ABS2_NUMBA_CACHE: dict[str, Any] = {}


def build_complex_abs2_module_spec() -> dict[str, Any]:
    src = "https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray"
    return {
        "name": "Memory-Efficient abs2 for complex NumPy arrays",
        "source_url": src,
        "goal": "Compute |z|^2 for large complex arrays with low temporary-memory pressure.",
        "notes": [
            "Use in-place operations (`out=`) where possible.",
            "If input must be preserved, prefer single-output pipelines.",
            "Numba vectorize backend can avoid large intermediate arrays.",
        ],
    }


def _estimate_abs2_peak_bytes(np_mod: Any, arr_np: Any, backend: str, preserve_input: bool) -> int:
    input_bytes = int(arr_np.nbytes)
    real_itemsize = int(arr_np.real.dtype.itemsize)
    output_bytes = int(arr_np.size * real_itemsize)
    backend = backend.lower()

    if backend == "numpy_inplace_complex":
        extra = 0 if not preserve_input else input_bytes
    elif backend == "conjugate_multiply":
        extra = input_bytes
    elif backend == "float_view_dot":
        try:
            is_contiguous = bool(arr_np.flags["C_CONTIGUOUS"])
        except Exception:  # noqa: BLE001
            is_contiguous = False
        contiguous_copy = 0 if is_contiguous else input_bytes
        extra = output_bytes + contiguous_copy
    else:
        extra = output_bytes

    return input_bytes + extra


def compute_complex_abs2(
    arr: Any,
    backend: str = "auto",
    preserve_input: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import importlib

    np_mod = importlib.import_module("numpy")
    arr_np = np_mod.asarray(arr)
    if not np_mod.iscomplexobj(arr_np):
        raise ValueError("compute_complex_abs2 expects a complex ndarray.")

    backend = backend.strip().lower()
    if backend not in {"auto", "numpy_abs_square", "numpy_inplace_complex", "numba_vectorize", "conjugate_multiply", "float_view_dot"}:
        raise ValueError("Unsupported backend for complex abs2 module.")

    if backend == "auto":
        if preserve_input:
            try:
                importlib.import_module("numba")
                backend = "numba_vectorize"
            except Exception:  # noqa: BLE001
                backend = "numpy_abs_square"
        else:
            backend = "numpy_inplace_complex"

    if backend == "numpy_abs_square":
        out = np_mod.abs(arr_np)
        np_mod.square(out, out=out)
    elif backend == "numpy_inplace_complex":
        work = arr_np.copy() if preserve_input else arr_np
        np_mod.abs(work, out=work)
        np_mod.square(work, out=work)
        out = work.real.copy()
    elif backend == "numba_vectorize":
        numba_mod = importlib.import_module("numba")
        dtype_key = str(arr_np.dtype)
        if dtype_key not in _ABS2_NUMBA_CACHE:
            if arr_np.dtype == np_mod.complex128:
                signature = [numba_mod.float64(numba_mod.complex128)]
            elif arr_np.dtype == np_mod.complex64:
                signature = [numba_mod.float32(numba_mod.complex64)]
            else:
                raise ValueError("numba_vectorize supports complex64 and complex128.")

            @numba_mod.vectorize(signature, nopython=True, cache=True)
            def _abs2_numba(x: Any) -> Any:
                return x.real * x.real + x.imag * x.imag

            _ABS2_NUMBA_CACHE[dtype_key] = _abs2_numba
        out = _ABS2_NUMBA_CACHE[dtype_key](arr_np)
    elif backend == "conjugate_multiply":
        work = arr_np.conjugate()
        np_mod.multiply(arr_np, work, out=work)
        out = work.real.copy()
    else:
        # float_view_dot
        if arr_np.dtype == np_mod.complex128:
            float_type = np_mod.float64
        elif arr_np.dtype == np_mod.complex64:
            float_type = np_mod.float32
        else:
            raise ValueError("float_view_dot supports complex64 and complex128.")
        src = arr_np if bool(arr_np.flags["C_CONTIGUOUS"]) else np_mod.ascontiguousarray(arr_np)
        float_view = src.view(float_type).reshape(arr_np.shape + (2,))
        out = np_mod.einsum("...i,...i->...", float_view, float_view, optimize=True)

    info = {
        "backend_used": backend,
        "input_dtype": str(arr_np.dtype),
        "output_dtype": str(out.dtype),
        "input_nbytes": int(arr_np.nbytes),
        "output_nbytes": int(out.nbytes),
        "peak_estimate_bytes": _estimate_abs2_peak_bytes(np_mod, arr_np, backend, preserve_input),
        "preserve_input_requested": bool(preserve_input),
        "source_url": build_complex_abs2_module_spec()["source_url"],
    }
    return out, info


def _parse_shape_csv(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Shape string is empty.")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError("All shape dimensions must be positive.")
    return dims


def build_realpython_complex_module() -> BuiltinComplexModuleRef:
    src = "https://realpython.com/ref/builtin-types/complex/"
    return BuiltinComplexModuleRef(
        name="Python built-in complex() reference module",
        short_description="Create and work with complex numbers using Python's built-in complex type.",
        last_verified="2026-03-13",
        constructor_syntax=[
            "complex(real=0, imag=0)",
            "complex(string)",
        ],
        arguments={
            "real": "Numeric real part when imag is provided.",
            "imag": "Numeric imaginary part when provided with real.",
            "string": "String representation like '1+2j' (imag must not be passed).",
        },
        return_value="A complex number object with `.real` and `.imag` components.",
        examples=[
            "complex(1, 2) -> (1+2j)",
            "complex('3+4j') -> (3+4j)",
            "z = 1+2j; z.conjugate() -> (1-2j)",
            "abs(3+4j) -> 5.0",
        ],
        methods={
            "conjugate()": "Return the complex conjugate (flip sign of imaginary part).",
            "real": "Read the real component.",
            "imag": "Read the imaginary component.",
            "__abs__ / abs(z)": "Return the magnitude sqrt(real^2 + imag^2).",
            "__add__, __sub__, __mul__, __truediv__": "Standard complex arithmetic operators.",
        },
        common_use_cases=[
            "Signal processing and electrical engineering calculations",
            "2D geometry representation and transforms",
            "Distance between points via abs(z1 - z2)",
            "Polar/cartesian reasoning in physics and simulations",
        ],
        real_world_example=[
            "Point A = 2+3j and Point B = 4+7j",
            "Distance = abs((4+7j) - (2+3j)) = abs(2+4j) = sqrt(20)",
            "This pattern maps naturally to coordinate distance calculations.",
        ],
        related_resources=[
            "https://realpython.com/python-complex-numbers/",
            "https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex",
        ],
        source_url=src,
    )


def _parse_complex_literal(value: str) -> complex:
    text = value.strip().replace(" ", "")
    if not text:
        raise ValueError("Complex literal is empty.")
    try:
        return complex(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid complex literal: {value}") from exc


def run_complex_module_demo(a_literal: str, b_literal: str) -> dict[str, Any]:
    a = _parse_complex_literal(a_literal)
    b = _parse_complex_literal(b_literal)
    midpoint = (a + b) / 2
    delta = b - a
    return {
        "input_a": str(a),
        "input_b": str(b),
        "sum": str(a + b),
        "difference_b_minus_a": str(delta),
        "product": str(a * b),
        "conjugate_a": str(a.conjugate()),
        "conjugate_b": str(b.conjugate()),
        "magnitude_a": float(abs(a)),
        "magnitude_b": float(abs(b)),
        "distance_between_points": float(abs(delta)),
        "midpoint": str(midpoint),
        "phase_alignment_hint": "Smaller absolute phase difference suggests better alignment in oscillatory systems.",
    }


def build_dasroot_emerging_python_2026_modules() -> list[EmergingPython2026ModuleRef]:
    src = "https://dasroot.net/posts/2026/03/emerging-python-use-cases-2026/"
    legal = [
        "Respect robots.txt and site terms before scraping or automated extraction.",
        "Store only minimum required data and avoid sensitive personal information by default.",
        "Prefer API/data-export endpoints when available over aggressive HTML crawling.",
    ]
    return [
        EmergingPython2026ModuleRef(
            name="NL Data Analysis Copilot",
            category="AI-driven data analysis",
            summary="Natural-language analytics workflow where users ask plain-language questions over tabular data.",
            python_stack=["Python", "pandas", "OpenAI API", "Jupyter"],
            suggested_application="Build a notebook/CLI copilot that converts text queries into data transforms and charts.",
            cited_versions={"article_date": "2026-03"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Anaconda LLM Experiment Lab",
            category="AI-driven data analysis",
            summary="Local-first experimentation with AI Navigator-style environment management for LLM workflows.",
            python_stack=["Python", "conda", "Transformers", "PyTorch"],
            suggested_application="Create a reproducible prompt/model comparison runner with environment snapshots.",
            cited_versions={"Anaconda AI Navigator": "2.0"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Jupyter AI Model Builder",
            category="AI-driven data analysis",
            summary="Notebook-native assistant to streamline feature prep, model training, and evaluation loops.",
            python_stack=["Python", "Jupyter AI", "scikit-learn", "matplotlib"],
            suggested_application="Autogenerate training/evaluation notebook cells from dataset metadata.",
            cited_versions={"Jupyter AI": "2.31.6"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Temporal Workflow Orchestrator",
            category="Distributed systems and automation",
            summary="Long-running, reliable orchestration using workflows, activities, and signals.",
            python_stack=["Python", "Temporal SDK", "PostgreSQL", "Redis"],
            suggested_application="Build fault-tolerant AI job pipelines with retries and human approval signals.",
            cited_versions={"Temporal Python SDK": "2.10.0"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Polars ETL Acceleration",
            category="Modern high-performance libraries",
            summary="Columnar DataFrame processing optimized for speed and low memory pressure.",
            python_stack=["Python", "Polars", "Arrow", "Parquet"],
            suggested_application="Replace pandas-heavy ingestion stages with Polars lazy pipelines.",
            cited_versions={"Polars": "0.19.0"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Smolagents Action Engine",
            category="Modern high-performance libraries",
            summary="Lightweight agent framework with tool calls and sandboxed execution patterns.",
            python_stack=["Python", "smolagents", "Transformers"],
            suggested_application="Create constrained autonomous task runners for data ops and report generation.",
            cited_versions={"smolagents": "1.2.0"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="MarkItDown Ingestion Gateway",
            category="Document processing",
            summary="Normalize diverse document types into markdown for LLM-friendly retrieval/indexing.",
            python_stack=["Python", "markitdown", "pathlib"],
            suggested_application="Build a document-to-markdown ingestion stage before vector indexing.",
            cited_versions={"MarkItDown": "2.1.0"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="PySide6 Desktop Ops Console",
            category="GUI and frontend",
            summary="Desktop-native monitoring/control surfaces for AI pipelines and model status.",
            python_stack=["Python", "PySide6", "Qt", "asyncio"],
            suggested_application="Ship an internal desktop console with realtime workflow status and controls.",
            cited_versions={"PySide6/PyQt6": "6.5.2"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="NiceGUI Rapid App Layer",
            category="GUI and frontend",
            summary="Fast web app interfaces for dashboards, controls, and simple workflow portals.",
            python_stack=["Python", "NiceGUI", "FastAPI"],
            suggested_application="Expose AI workflow controls and metrics as lightweight internal web apps.",
            cited_versions={"NiceGUI": "latest in article"},
            compliance_notes=legal,
            source_url=src,
        ),
        EmergingPython2026ModuleRef(
            name="Streamlit Insight Workbench",
            category="GUI and frontend",
            summary="Low-friction analytical frontends for model outputs and business insight sharing.",
            python_stack=["Python", "Streamlit", "pandas", "plotly"],
            suggested_application="Deliver stakeholder-facing analytics views with minimal frontend overhead.",
            cited_versions={"Streamlit": "latest in article"},
            compliance_notes=legal,
            source_url=src,
        ),
    ]


def build_linkedin_python_trends_2025_modules() -> list[LinkedInPython2025TrendModuleRef]:
    src = "https://www.linkedin.com/pulse/top-python-trends-2025-what-every-developer-should-know-goud-0cvbcf"
    return [
        LinkedInPython2025TrendModuleRef(
            trend_id=1,
            trend_name="AI and Machine Learning Dominance",
            summary="Python remains central for ML/AI workflows from experimentation to deployment.",
            python_stack=["Python", "PyTorch", "TensorFlow", "scikit-learn"],
            suggested_application="Add an AI experiment orchestration module with standardized train/eval pipelines.",
            learning_path=["Linear algebra and statistics", "Model training", "MLOps deployment basics"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=2,
            trend_name="Data Science and Big Data Evolution",
            summary="Data-heavy workflows continue to scale with Python analytics and pipeline tools.",
            python_stack=["Python", "pandas", "Polars", "Dask", "Apache Spark"],
            suggested_application="Build a dual-engine ETL app that auto-switches between pandas and Polars/Dask by dataset size.",
            learning_path=["Data wrangling", "Feature engineering", "Distributed processing"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=3,
            trend_name="Web Development with Modern Frameworks",
            summary="FastAPI and Django-style ecosystems remain strong for service and product backends.",
            python_stack=["Python", "FastAPI", "Django", "SQLModel"],
            suggested_application="Add a microservice generator for API + auth + telemetry bootstrap.",
            learning_path=["HTTP and REST", "Async Python", "Backend architecture patterns"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=4,
            trend_name="Cybersecurity and Ethical Hacking Growth",
            summary="Python tooling is widely used for security testing, monitoring, and analysis automation.",
            python_stack=["Python", "Scapy", "Requests", "YARA"],
            suggested_application="Create a compliance-first security scanner module with safe read-only probes.",
            learning_path=["Network fundamentals", "Security testing ethics", "Secure coding practices"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=5,
            trend_name="Cloud Computing and Serverless Expansion",
            summary="Python powers cloud-native automation and function-driven workloads across providers.",
            python_stack=["Python", "AWS Lambda", "Azure Functions", "Google Cloud Functions"],
            suggested_application="Add a deployment planner that maps workloads to serverless vs container runtimes.",
            learning_path=["Cloud architecture", "IaC basics", "Cost/performance tuning"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=6,
            trend_name="Automation and DevOps Integration",
            summary="Automation remains a major Python use case in CI/CD and platform operations.",
            python_stack=["Python", "GitHub Actions", "Ansible", "Docker SDK"],
            suggested_application="Implement a pipeline auto-remediator for failed CI steps with playbook hints.",
            learning_path=["Scripting fundamentals", "CI/CD pipelines", "Observability"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=7,
            trend_name="IoT and Edge Computing Adoption",
            summary="Python continues to be practical for prototyping and deploying edge intelligence.",
            python_stack=["Python", "MicroPython", "CircuitPython", "MQTT"],
            suggested_application="Add an edge telemetry module with offline buffering and delayed sync.",
            learning_path=["Embedded basics", "Sensor data handling", "Edge model optimization"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=8,
            trend_name="Blockchain and Web3 Prototyping",
            summary="Python is used for blockchain analytics, smart-contract tooling, and test harnesses.",
            python_stack=["Python", "web3.py", "Brownie", "eth-account"],
            suggested_application="Build a chain-observer app for transaction risk and anomaly summaries.",
            learning_path=["Blockchain fundamentals", "Smart contract interaction", "Security auditing basics"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=9,
            trend_name="Quantum Computing Exploration",
            summary="Python remains the entry language for quantum experimentation and simulators.",
            python_stack=["Python", "Qiskit", "Cirq", "NumPy"],
            suggested_application="Create a hybrid classical-quantum simulator runner for education and benchmarking.",
            learning_path=["Linear algebra", "Quantum gates/circuits", "Simulator interpretation"],
            source_url=src,
        ),
        LinkedInPython2025TrendModuleRef(
            trend_id=10,
            trend_name="Low-Code/No-Code API Automation",
            summary="Python bridges enterprise no-code tools with custom logic and integrations.",
            python_stack=["Python", "FastAPI", "Zapier/Make webhooks", "Pydantic"],
            suggested_application="Add a no-code connector module that validates payloads and executes safe automations.",
            learning_path=["API design", "Webhook security", "Integration testing"],
            source_url=src,
        ),
    ]


def build_vr_autonomy_accessibility_modules() -> list[VRAutonomyAccessibilityModuleRef]:
    nlp_mental_src = "https://www.nimh.nih.gov/health/topics/technology-and-the-future-of-mental-health-treatment"
    nasa_tlx_src = "https://humansystems.arc.nasa.gov/groups/TLX/"
    return [
        VRAutonomyAccessibilityModuleRef(
            module_id=1,
            name="VR Automation Control Plane",
            category="automation system",
            summary="Use Python workflow orchestration to schedule world tasks, retries, and safety approvals.",
            python_stack=["Python", "Prefect", "FastAPI", "Pydantic", "PostgreSQL"],
            mixed_stack=["Rust workers (optional)", "C++ device bridge adapters (optional)"],
            target_outcome="Reliable autonomous job execution for multi-scene VR pipelines.",
            compliance_notes=[
                "Require explicit approval gates for high-impact world-state changes.",
                "Maintain audit logs for autonomous decisions and rollback actions.",
            ],
            sources=["https://github.com/PrefectHQ/prefect"],
        ),
        VRAutonomyAccessibilityModuleRef(
            module_id=2,
            name="Auto-Update Guardrail Engine",
            category="auto update",
            summary="Automate dependency and module updates with staged rollout, policy checks, and fast rollback.",
            python_stack=["Python", "Dependabot", "Renovate", "pytest", "pip-tools"],
            mixed_stack=["GitHub Actions YAML", "Container image scanning tools"],
            target_outcome="Continuously updated runtime with lower breakage risk.",
            compliance_notes=[
                "Pin versions and verify signatures/checksums before deployment.",
                "Ship canary updates first, then promote only after health checks pass.",
            ],
            sources=[
                "https://docs.github.com/en/code-security/dependabot/dependabot-version-updates",
                "https://docs.renovatebot.com/",
            ],
        ),
        VRAutonomyAccessibilityModuleRef(
            module_id=3,
            name="VR World Autopilot Navigator",
            category="autopilot for vr worlds",
            summary="Train/predict navigation actions that keep users safe while moving through complex virtual scenes.",
            python_stack=["Python", "Habitat-Lab", "PyTorch", "NumPy"],
            mixed_stack=["C++ simulation backends", "Unity/C# world runtime (optional)"],
            target_outcome="Autonomous path guidance with obstacle avoidance in VR worlds.",
            compliance_notes=[
                "Keep a human override and emergency stop at all times.",
                "Constrain autopilot velocity/acceleration to comfort-safe VR thresholds.",
            ],
            sources=["https://github.com/facebookresearch/habitat-lab"],
        ),
        VRAutonomyAccessibilityModuleRef(
            module_id=4,
            name="Haptic Guidance for Blind and Low-Vision Users",
            category="haptic enhancement for blind users",
            summary="Convert scene hazards, route direction, and interaction affordances into structured haptic cues.",
            python_stack=["Python", "OpenXR bindings", "NumPy", "asyncio"],
            mixed_stack=["OpenXR C API", "C++ haptic driver plugins"],
            target_outcome="Accessible non-visual navigation and interaction in VR.",
            compliance_notes=[
                "Allow user-tunable vibration strength and pattern profiles.",
                "Provide redundant cues (haptic + audio) for critical safety events.",
            ],
            sources=[
                "https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPath.html",
                "https://www.microsoft.com/en-us/research/publication/haptic-and-auditory-cane-for-virtual-environment/",
            ],
        ),
        VRAutonomyAccessibilityModuleRef(
            module_id=5,
            name="VR Hearing Assist and Audio Caption Layer",
            category="better hearing for vr",
            summary="Add real-time captions and directional audio cues to improve speech and sound-event comprehension.",
            python_stack=["Python", "Whisper", "WebVTT", "FastAPI"],
            mixed_stack=["C++ spatial audio engine", "WebXR/WebAudio overlays"],
            target_outcome="Improved accessibility for d/Deaf and hard-of-hearing VR users.",
            compliance_notes=[
                "Captions should be synchronized, accurate, and available for all key audio content.",
                "Include speech/non-speech context labels for critical environmental sounds.",
            ],
            sources=[
                "https://www.w3.org/WAI/media/av/captions/",
                "https://www.researchgate.net/publication/318430895_Sound_Accessibility_in_Virtual_Reality_Using_Interactive_3D_Audio_Captions",
            ],
        ),
        VRAutonomyAccessibilityModuleRef(
            module_id=6,
            name="Mental Mode Regulator",
            category="mental mode",
            summary="Estimate cognitive load/stress and adapt interaction pace, difficulty, and interruption policies in-session.",
            python_stack=["Python", "NumPy", "pandas", "scikit-learn", "FastAPI"],
            mixed_stack=["Wearable SDKs", "Unity/C# adaptive UI overlays"],
            target_outcome="Lower overload risk and more stable long-session performance.",
            compliance_notes=[
                "This is wellness support and workload adaptation, not medical diagnosis.",
                "Offer local emergency-resource escalation pathways for severe distress signals.",
            ],
            sources=[nlp_mental_src, nasa_tlx_src],
        ),
    ]


def build_openai_module_catalog() -> list[OpenAIModuleRef]:
    # Curated from official OpenAI documentation.
    return [
        OpenAIModuleRef(
            name="Agents SDK",
            url="https://platform.openai.com/docs/guides/agents-sdk",
            focus="Agent orchestration, tool use, handoffs, and traces",
            why_it_matters="Fastest official path to robust multi-agent workflows.",
        ),
        OpenAIModuleRef(
            name="Responses API Tools",
            url="https://platform.openai.com/docs/guides/tools",
            focus="Web search, file search, function calling, remote MCP",
            why_it_matters="Adds external action capability without custom middleware first.",
        ),
        OpenAIModuleRef(
            name="Agent Evals",
            url="https://platform.openai.com/docs/guides/agent-evals",
            focus="Reproducible quality measurement for agent workflows",
            why_it_matters="Creates an optimization flywheel to prevent regressions.",
        ),
        OpenAIModuleRef(
            name="Trace Grading",
            url="https://platform.openai.com/docs/guides/trace-grading",
            focus="Step-level grading across traces and runs",
            why_it_matters="Pinpoints failure stages for targeted fixes and safer iteration.",
        ),
    ]


def build_github_module_catalog() -> list[GitHubModuleRef]:
    # Curated from live GitHub scan (memory, reasoning, self-improvement).
    return [
        GitHubModuleRef(
            name="LangMem",
            repo="langchain-ai/langmem",
            url="https://github.com/langchain-ai/langmem",
            focus="Background memory management + prompt optimization",
            why_it_matters="Adds agent learning loops that refine behavior over time.",
        ),
        GitHubModuleRef(
            name="A-MEM",
            repo="agiresearch/A-mem",
            url="https://github.com/agiresearch/A-mem",
            focus="Agentic memory graph + evolving links",
            why_it_matters="Improves long-horizon recall and relationship-aware retrieval.",
        ),
        GitHubModuleRef(
            name="ReMe",
            repo="agentscope-ai/ReMe",
            url="https://github.com/agentscope-ai/ReMe",
            focus="Personal/task/tool memory separation",
            why_it_matters="Clean memory routing reduces context noise in complex runs.",
        ),
        GitHubModuleRef(
            name="General Agentic Memory",
            repo="VectorSpaceLab/general-agentic-memory",
            url="https://github.com/VectorSpaceLab/general-agentic-memory",
            focus="JIT memory retrieval and synthesis",
            why_it_matters="Supports adaptive context at runtime instead of static memory fetch.",
        ),
        GitHubModuleRef(
            name="MemEngine",
            repo="nuster1128/MemEngine",
            url="https://github.com/nuster1128/MemEngine",
            focus="Modular memory architecture for LLM agents",
            why_it_matters="Useful for plug-and-play memory backends and experimentation.",
        ),
        GitHubModuleRef(
            name="OpenMemory",
            repo="CaviraOSS/OpenMemory",
            url="https://github.com/CaviraOSS/OpenMemory",
            focus="Standalone memory OS with decay and graph recall",
            why_it_matters="Strong base for persistent memory with explainable trace behavior.",
        ),
    ]


def build_future_modules() -> list[FutureModule]:
    # Inspired by recent Google/DeepMind research patterns.
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
            description="AlphaEvolve-style iterative mutate-score-select loop for code and plans.",
            weight=1.15,
            heat_cost=1.4,
            energy_gain=0.8,
            stability_gain=0.7,
            innovation_gain=1.6,
        ),
    ]


def simulate_future_system(
    config: AgentConfig,
    steps: int = 12,
    modules: list[FutureModule] | None = None,
) -> SimulationResult:
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

        # Passive cooldown each step.
        perf.heat = max(0.0, perf.heat - (1.8 + perf.heat_resistance * 2.0))

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


def simulate_vr_resilience(
    config: AgentConfig,
    steps: int = 10,
    vr_modules: list[VRModuleRef] | None = None,
) -> VRSimulationResult:
    vr_modules = vr_modules or build_vr_module_catalog()
    perf = PerformanceState(
        writing_speed=max(0.6, config.writing_speed),
        activity_level=max(0.7, config.activity_level),
        energy=max(10.0, config.base_energy),
        heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
        burn_rate=max(1.0, config.burn_rate),
        heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
    )
    resilience = ImpactResilienceState(
        resilience=max(0.35, config.resilience_factor),
        continuity=max(0.0, min(100.0, config.continuity_target)),
    )

    foresight = 0.0
    avoid_success = 0.0
    for turn in range(1, max(1, steps) + 1):
        temp = perf.tune_for_turn(config.temperature, turn, steps)
        horizon_glimpse = (perf.energy / 240.0) * (1.15 - min(1.0, perf.heat / 120.0))
        foresight += max(0.0, horizon_glimpse)

        predicted_risk = max(0.0, (1.0 - horizon_glimpse) + (temp * 0.25))
        event_severity = predicted_risk * (0.8 + ((turn % 3) * 0.1))
        resilience.ingest_event(event_severity, emotional_weight=1.0 + (temp * 0.5))
        resilience.recover(perf, forgetting_rate=max(0.05, config.negative_forgetting_rate))

        # Chance-like deterministic score: better foresight + resilience means more avoided impacts.
        avoid_success += max(0.0, 1.0 - event_severity + (resilience.resilience * 0.15))

    norm = float(max(1, steps))
    return VRSimulationResult(
        steps=max(1, steps),
        foresight_score=foresight / norm,
        avoidance_success_rate=min(1.0, max(0.0, avoid_success / norm)),
        pain_resilience_score=max(0.0, 1.0 - min(1.0, resilience.pain_load / 12.0)),
        continuity_score=resilience.continuity / 100.0,
        negative_memory_pressure=resilience.negative_memory_pressure,
        recommended_stack=[f"{m.name} ({m.language_mix})" for m in vr_modules[:3]],
    )


def simulate_vr_timetravel(
    config: AgentConfig,
    steps: int = 12,
    modules: list[VRTimeTravelModuleRef] | None = None,
) -> VRTimeTravelSimulationResult:
    modules = modules or build_semanticproxy_vr_timetravel_modules()
    perf = PerformanceState(
        writing_speed=max(0.6, config.writing_speed),
        activity_level=max(0.7, config.activity_level),
        energy=max(10.0, config.base_energy),
        heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
        burn_rate=max(1.0, config.burn_rate),
        heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
    )

    glimpse_total = 0.0
    avoid_total = 0.0
    frame_stability_total = 0.0
    latency_risk_total = 0.0

    module_boost = 0.03 * len(modules)
    for turn in range(1, max(1, steps) + 1):
        temp = perf.tune_for_turn(config.temperature, turn, steps)
        thermal_penalty = min(0.55, perf.heat / 120.0)
        base_glimpse = max(0.0, (perf.energy / 220.0) + module_boost - thermal_penalty)
        glimpse_total += base_glimpse

        predicted_hazard = max(0.0, 0.85 - base_glimpse + (temp * 0.2))
        avoid_total += max(0.0, 1.0 - predicted_hazard)

        frame_stability = max(0.0, 1.0 - (thermal_penalty * 0.9))
        frame_stability_total += frame_stability

        latency_risk = max(0.0, (temp * 0.35) + (thermal_penalty * 0.8) - (module_boost * 0.6))
        latency_risk_total += latency_risk

        # Model extra load from timeline branch simulation.
        perf.heat += 0.9 * (1.0 + latency_risk)
        perf.energy = max(0.0, perf.energy - (1.4 * (1.0 + predicted_hazard)))

    denom = float(max(1, steps))
    return VRTimeTravelSimulationResult(
        steps=max(1, steps),
        timeline_glimpse_score=glimpse_total / denom,
        hazard_avoidance_score=avoid_total / denom,
        frame_stability_score=frame_stability_total / denom,
        latency_risk=latency_risk_total / denom,
        recommended_modules=[m.name for m in modules[:4]],
    )


def simulate_vr_autonomy_accessibility(
    config: AgentConfig,
    steps: int = 14,
    modules: list[VRAutonomyAccessibilityModuleRef] | None = None,
) -> VRAutonomySimulationResult:
    modules = modules or build_vr_autonomy_accessibility_modules()
    perf = PerformanceState(
        writing_speed=max(0.6, config.writing_speed),
        activity_level=max(0.7, config.activity_level),
        energy=max(10.0, config.base_energy),
        heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
        burn_rate=max(1.0, config.burn_rate),
        heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
    )

    automation_total = 0.0
    update_total = 0.0
    autopilot_total = 0.0
    haptic_total = 0.0
    hearing_total = 0.0
    mental_total = 0.0

    module_boost = min(0.38, len(modules) * 0.026)
    for turn in range(1, max(1, steps) + 1):
        temp = perf.tune_for_turn(config.temperature, turn, steps)
        thermal_penalty = min(0.62, perf.heat / 135.0)
        energy_norm = min(1.2, max(0.0, perf.energy / 170.0))

        automation_score = max(
            0.0,
            min(
                1.0,
                0.66 + module_boost + (energy_norm * 0.14) - (thermal_penalty * 0.52) + (perf.activity_level * 0.04),
            ),
        )
        update_score = max(
            0.0,
            min(
                1.0,
                0.64 + module_boost + (config.heat_resistance * 0.12) - (temp * 0.24) - (thermal_penalty * 0.2),
            ),
        )
        autopilot_score = max(
            0.0,
            min(
                1.0,
                0.61 + module_boost + (energy_norm * 0.18) - (thermal_penalty * 0.55) - (temp * 0.18),
            ),
        )
        haptic_score = max(
            0.0,
            min(
                1.0,
                0.67 + module_boost + (config.resilience_factor * 0.08) - (thermal_penalty * 0.18),
            ),
        )
        hearing_score = max(
            0.0,
            min(
                1.0,
                0.66 + module_boost + (energy_norm * 0.09) - (temp * 0.12) - (thermal_penalty * 0.16),
            ),
        )
        mental_score = max(
            0.0,
            min(
                1.0,
                0.63 + module_boost + (config.resilience_factor * 0.14) - (temp * 0.23) - (thermal_penalty * 0.2),
            ),
        )

        automation_total += automation_score
        update_total += update_score
        autopilot_total += autopilot_score
        haptic_total += haptic_score
        hearing_total += hearing_score
        mental_total += mental_score

        # Model compute load and adaptive cooling under sustained autonomous control.
        perf.heat += 1.05 + (temp * 0.75) + ((1.0 - automation_score) * 0.5) + ((1.0 - autopilot_score) * 0.7)
        perf.energy = max(0.0, min(320.0, perf.energy + (haptic_score * 0.55) + (hearing_score * 0.5) - (temp * 5.5)))

    denom = float(max(1, steps))
    automation_reliability = automation_total / denom
    update_safety_score = update_total / denom
    autopilot_world_stability = autopilot_total / denom
    haptic_accessibility_score = haptic_total / denom
    hearing_accessibility_score = hearing_total / denom
    mental_mode_safety_score = mental_total / denom
    overall_safety_score = (
        (automation_reliability * 0.2)
        + (update_safety_score * 0.16)
        + (autopilot_world_stability * 0.2)
        + (haptic_accessibility_score * 0.16)
        + (hearing_accessibility_score * 0.16)
        + (mental_mode_safety_score * 0.12)
    )

    return VRAutonomySimulationResult(
        steps=max(1, steps),
        automation_reliability=automation_reliability,
        update_safety_score=update_safety_score,
        autopilot_world_stability=autopilot_world_stability,
        haptic_accessibility_score=haptic_accessibility_score,
        hearing_accessibility_score=hearing_accessibility_score,
        mental_mode_safety_score=mental_mode_safety_score,
        overall_safety_score=overall_safety_score,
    )


# ============================================================
# Memory
# ============================================================

class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
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
        self._ensure_column("memory", "importance", "REAL DEFAULT 0.5")
        self._ensure_column("memory", "last_accessed", "TEXT DEFAULT ''")
        self._ensure_column("memory", "access_count", "INTEGER DEFAULT 0")
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
            # FTS5 might be unavailable in some builds. The agent still works.
            pass

        self.conn.commit()

    def _ensure_column(self, table: str, col: str, col_def: str) -> None:
        try:
            cols = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            names = {c[1] for c in cols}
            if col not in names:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
        except sqlite3.Error:
            pass

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
        # Ranking blend inspired by modern retrieval systems: relevance + recency + importance + usage.
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
            overlap = sum(1 for t in tokens if t and t in content)
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

        scored.sort(key=lambda s: s[0], reverse=True)
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
    state: str = "pending"  # pending, ready, running, done, blocked
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
    payload = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    return json.dumps(payload, indent=2)


def _safe_regex_sub(pattern: str, repl: str, text: str, count: int = 0) -> tuple[str, int]:
    compiled = re.compile(pattern, flags=re.MULTILINE | re.DOTALL)
    return compiled.subn(repl, text, count=count)


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
        # Responses API function tool format
        return [
            {
                "type": "function",
                "name": "list_files",
                "description": "List files and directories under a path in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
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
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer"},
                    },
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
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
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
                    "properties": {
                        "code": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
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
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
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
                    "properties": {
                        "path": {"type": "string"},
                    },
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
                    "properties": {
                        "path": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
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
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
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
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["url"],
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
    pattern = re.compile(args["pattern"])
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
        return json.dumps({"path": str(path.relative_to(ctx.workspace)), "ok": True}, indent=2)
    except SyntaxError as exc:
        return json.dumps(
            {
                "path": str(path.relative_to(ctx.workspace)),
                "ok": False,
                "line": exc.lineno,
                "offset": exc.offset,
                "msg": exc.msg,
                "text": exc.text,
            },
            indent=2,
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
    limit = int(args.get("limit", 5))
    endpoint = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)

    req = urllib.request.Request(endpoint, headers={"User-Agent": f"{VERSION}/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")

    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)">(?P<title>.*?)</a>',
        re.IGNORECASE,
    )
    hits = []
    for m in pattern.finditer(html):
        title = re.sub("<.*?>", "", m.group("title"))
        href = m.group("href")
        hits.append({"title": title, "url": href})
        if len(hits) >= limit:
            break
    return json.dumps({"query": query, "results": hits}, indent=2)


def tool_fetch_url(ctx: ToolContext, args: dict[str, Any]) -> str:
    url = args["url"]
    max_chars = int(args.get("max_chars", 8000))
    req = urllib.request.Request(url, headers={"User-Agent": f"{VERSION}/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8", errors="replace")

    text = re.sub(r"<script.*?</script>", "", body, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


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
            except Exception as exc:  # noqa: BLE001
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

        # Common convenience fields
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

        return ModelTurn(
            content="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            raw=raw,
        )

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
- Maintain high activity and energetic but clear writing.
- End with a final answer only when you have enough evidence.
"""

GOOGLE_INSPIRED_IDEAS = [
    "Iterative planning with fast feedback loops (DeepMind agent practice).",
    "Retrieval-first memory grounding before major decisions.",
    "Self-critique checkpoints to reduce drift and hallucination.",
    "Tool orchestration from explicit objectives.",
]

VR_OUTLOOK_METHODS = [
    "Short-horizon predictive control (Python: NumPy/JAX; mixed: C++ solver via pybind11).",
    "Uncertainty-aware risk scoring with conformal calibration (Python-first).",
    "Simulation policy training with Unity ML-Agents (Python trainer + C# runtime).",
    "Low-latency inference path with ONNX Runtime (Python orchestration + native backend).",
    "SemanticProxy stack fit: OpenVR + PyOpenGL + Pygame + asyncio + OpenCV/TensorFlow.",
]


class CodexUnifiedAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.memory = MemoryStore(config.memory_db_path)
        self.tools = ToolRegistry(ToolContext(config.workspace, config.max_tool_output_chars))
        self.client = OpenAIClient(config)
        self.task_graph = TaskGraph()
        self.future_modules = build_future_modules()
        self.github_modules = build_github_module_catalog()
        self.openai_modules = build_openai_module_catalog()
        self.vr_modules = build_vr_module_catalog()
        self.vr_timetravel_modules = build_semanticproxy_vr_timetravel_modules()
        self.dasroot_2026_modules = build_dasroot_emerging_python_2026_modules()
        self.linkedin_2025_modules = build_linkedin_python_trends_2025_modules()
        self.vr_autonomy_accessibility_modules = build_vr_autonomy_accessibility_modules()
        self.web_idea_catalog = build_bestprojectideas_ai50_catalog()
        self.requested_addons = build_requested_addons32_catalog()
        self.performance = PerformanceState(
            writing_speed=max(0.6, config.writing_speed),
            activity_level=max(0.7, config.activity_level),
            energy=max(10.0, config.base_energy),
            heat_resistance=max(0.0, min(0.99, config.heat_resistance)),
            burn_rate=max(1.0, config.burn_rate),
            heat_absorb_rate=max(0.05, min(0.95, config.heat_absorb_rate)),
        )
        self.resilience_state = ImpactResilienceState(
            resilience=max(0.35, config.resilience_factor),
            continuity=max(0.0, min(100.0, config.continuity_target)),
        )

    def close(self) -> None:
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
        module_lines = "\n".join(f"- {m.name}: {m.description}" for m in self.future_modules)
        github_lines = "\n".join(
            f"- {m.name} [{m.repo}] {m.focus} :: {m.url}" for m in self.github_modules[:4]
        )
        openai_lines = "\n".join(
            f"- {m.name}: {m.focus} :: {m.url}" for m in self.openai_modules
        )
        vr_lines = "\n".join(
            f"- {m.name}: {m.focus} ({m.language_mix}) :: {m.url}" for m in self.vr_modules[:4]
        )
        vr_tt_lines = "\n".join(
            f"- {m.name}: {m.focus} [{', '.join(m.python_stack[:2])}] :: {m.source_url}"
            for m in self.vr_timetravel_modules[:4]
        )
        dasroot_lines = "\n".join(
            f"- {m.name} ({m.category}) :: {m.suggested_application}"
            for m in self.dasroot_2026_modules[:6]
        )
        linkedin_lines = "\n".join(
            f"- #{m.trend_id} {m.trend_name} :: {m.suggested_application}"
            for m in self.linkedin_2025_modules[:6]
        )
        vr_autonomy_lines = "\n".join(
            f"- #{m.module_id} {m.name} ({m.category}) :: {m.target_outcome}"
            for m in self.vr_autonomy_accessibility_modules[:6]
        )
        idea_sample = "\n".join(
            f"- #{i.idea_id} [{i.level}] {i.title} ({', '.join(i.languages[:2])})"
            for i in self.web_idea_catalog[:10]
        )
        addon_sample = "\n".join(
            f"- #{a.addon_id} {a.title} [{', '.join(a.primary_python_stack[:3])}]"
            for a in self.requested_addons[:12]
        )
        nsp_hint = "- Next Sentence Prediction module is included (CLI: --show-nsp-module / --nsp-predict)."
        abs2_hint = "- Complex abs2 memory-efficient module is included (CLI: --show-abs2-module / --abs2-demo)."
        complex_builtin_hint = "- Built-in complex() module is included (CLI: --show-complex-module / --complex-demo)."
        vr_autonomy_hint = "- VR autonomy/accessibility modules are included (CLI: --list-vr-autonomy-modules / --simulate-vr-autonomy)."
        vr_methods = "\n".join(f"- {m}" for m in VR_OUTLOOK_METHODS)
        return (
            f"Google-inspired strategy:\n{ideas}\n\n"
            f"Future modules:\n{module_lines}\n\n"
            f"OpenAI module candidates:\n{openai_lines}\n\n"
            f"GitHub module candidates:\n{github_lines}\n\n"
            f"VR foresight + resilience modules:\n{vr_lines}\n\n"
            f"VR time-travel modules (SemanticProxy sourced):\n{vr_tt_lines}\n\n"
            f"Emerging Python 2026 modules (DasRoot sourced):\n{dasroot_lines}\n\n"
            f"Python Trends 2025 applications (LinkedIn sourced):\n{linkedin_lines}\n\n"
            f"VR autonomy + accessibility modules:\n{vr_autonomy_lines}\n\n"
            f"BestProjectIdeas AI catalog sample (10/50):\n{idea_sample}\n\n"
            f"Requested Python add-ons sample (12/32):\n{addon_sample}\n\n"
            f"NSP module:\n{nsp_hint}\n\n"
            f"Complex abs2 module:\n{abs2_hint}\n\n"
            f"Built-in complex module:\n{complex_builtin_hint}\n\n"
            f"VR autonomy/accessibility module:\n{vr_autonomy_hint}\n\n"
            f"VR implementation paths (Python / mixed-language):\n{vr_methods}\n\n"
            f"Remembrance guide:\n{remembrance}"
        )

    def _seed_task_graph(self, goal: str) -> None:
        self.task_graph.add("inspect", "Inspect the workspace and understand code structure.")
        self.task_graph.add("plan", f"Plan execution for goal: {goal}", deps=["inspect"])
        self.task_graph.add("implement", "Apply code changes and improvements.", deps=["plan"])
        self.task_graph.add("validate", "Run syntax checks and tests when possible.", deps=["implement"])
        self.task_graph.add("review", "Critique and patch remaining issues.", deps=["validate"])

    def run(self, goal: str) -> AgentRunResult:
        self._seed_task_graph(goal)
        mem_ctx = self._memory_context(goal)
        innovation_guide = self._innovation_guide(goal)

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
                    f"Performance state: {self.performance.snapshot()}\n"
                    f"Resilience state: {self.resilience_state.snapshot()}\n"
                    "Start by inspecting the workspace, then make a concise plan, then execute it with momentum."
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
            synthetic_event = max(0.0, (turn_temp * 0.45) + (self.performance.heat / 130.0))
            self.resilience_state.ingest_event(synthetic_event, emotional_weight=1.0 + (turn_temp * 0.5))
            self.resilience_state.recover(self.performance, forgetting_rate=self.config.negative_forgetting_rate)
            self._log(
                f"turn={turn} previous_response_id={previous_response_id!r} "
                f"temperature={turn_temp:.2f} perf={self.performance.snapshot()} resilience={self.resilience_state.snapshot()}"
            )

            if turn > 1 and turn % 2 == 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Remembrance refresh: keep high activity, concise writing, and grounded tool use. "
                            f"Current performance: {self.performance.snapshot()}. "
                            f"Current resilience: {self.resilience_state.snapshot()}"
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
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool calls detected. Continue with concrete execution or give final output if done.",
                    }
                )
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
                    except Exception as exc:  # noqa: BLE001
                        tool_failures[call.name] += 1
                        tool_out = f"Tool error ({call.name}): {exc}\n{traceback.format_exc(limit=2)}"

                self._log(f"tool={call.name} repeated={repeated_calls[key]} failures={tool_failures[call.name]}")
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": tool_out,
                    }
                )

                # For chat-completions fallback compatibility, keep a transcript too.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or call.call_id,
                        "content": tool_out,
                    }
                )

            function_outputs = outputs

            if turn == 1:
                self.task_graph.mark_done("inspect", "Workspace inspection performed by the model/tools.")
                self.task_graph.mark_done("plan", "Initial plan produced.")
            elif turn == 2:
                self.task_graph.mark_done("implement", "Implementation phase active.")
            elif turn >= 3:
                self.task_graph.mark_done("validate", "Validation phase reached.")

            if turn % self.config.reviewer_every == 0:
                softened = self.memory.soften_negative_memories(self.config.negative_forgetting_rate)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reviewer phase: identify flaws, risky assumptions, syntax/runtime issues, "
                            "and missing validations. Patch them before finalizing. "
                            f"Applied negative-memory attenuation to {softened} records."
                        ),
                    }
                )

        self.task_graph.mark_done("review", "Review completed or loop ended.")

        self.memory.add(MemoryItem(key="goal", value=goal, tags="run", importance=0.85))
        self.memory.add(
            MemoryItem(
                key="result",
                value=truncate(final_answer or "No final answer produced.", 1200),
                tags="run",
                importance=0.9,
            )
        )
        self.memory.add(
            MemoryItem(
                key="guide",
                value=truncate(innovation_guide, 1200),
                tags="remembrance,google-inspired",
                importance=0.7,
            )
        )
        self.memory.add(
            MemoryItem(
                key="resilience",
                value=truncate(self.resilience_state.snapshot(), 1000),
                tags="run,resilience,continuity",
                importance=0.75,
            )
        )

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
    parser.add_argument("--resilience-factor", type=float, default=None, help="Override resilience factor")
    parser.add_argument("--negative-forgetting-rate", type=float, default=None, help="Override negative-memory forgetting rate")
    parser.add_argument("--simulate", action="store_true", help="Run future-module simulation and exit")
    parser.add_argument("--simulation-steps", type=int, default=12, help="Simulation step count")
    parser.add_argument("--simulate-vr", action="store_true", help="Run VR foresight/resilience simulation and exit")
    parser.add_argument("--vr-steps", type=int, default=10, help="VR simulation step count")
    parser.add_argument("--simulate-vr-timetravel", action="store_true", help="Run VR time-travel simulation and exit")
    parser.add_argument("--vr-timetravel-steps", type=int, default=12, help="VR time-travel simulation step count")
    parser.add_argument("--simulate-vr-autonomy", action="store_true", help="Run VR autonomy/accessibility simulation and exit")
    parser.add_argument("--vr-autonomy-steps", type=int, default=14, help="VR autonomy/accessibility simulation step count")
    parser.add_argument("--list-github-modules", action="store_true", help="List curated GitHub module catalog and exit")
    parser.add_argument("--list-openai-modules", action="store_true", help="List curated OpenAI module catalog and exit")
    parser.add_argument("--list-dasroot-2026-modules", action="store_true", help="List DasRoot 2026 emerging Python modules and exit")
    parser.add_argument("--list-linkedin-python2025-modules", action="store_true", help="List LinkedIn Python Trends 2025 application modules and exit")
    parser.add_argument("--list-vr-modules", action="store_true", help="List curated VR module catalog and exit")
    parser.add_argument("--list-vr-timetravel-modules", action="store_true", help="List SemanticProxy-based VR time-travel modules and exit")
    parser.add_argument("--list-vr-autonomy-modules", action="store_true", help="List VR autonomy/accessibility module catalog and exit")
    parser.add_argument("--list-bestprojectideas-ai50", action="store_true", help="List all 50 AI ideas sourced from BestProjectIdeas and exit")
    parser.add_argument("--list-requested-addons32", action="store_true", help="List all 32 user-requested Python add-ons and exit")
    parser.add_argument("--show-nsp-module", action="store_true", help="Show Next Sentence Prediction (BERT) module spec and exit")
    parser.add_argument("--nsp-predict", action="store_true", help="Run Next Sentence Prediction on --sentence-a/--sentence-b and exit")
    parser.add_argument("--sentence-a", default="", help="First sentence for NSP")
    parser.add_argument("--sentence-b", default="", help="Second sentence for NSP")
    parser.add_argument("--nsp-model", default="bert-base-uncased", help="Transformers model id for NSP")
    parser.add_argument("--nsp-backend", default="auto", choices=["auto", "transformers", "heuristic"], help="NSP backend")
    parser.add_argument("--show-abs2-module", action="store_true", help="Show memory-efficient complex abs2 module spec and exit")
    parser.add_argument("--abs2-demo", action="store_true", help="Run complex abs2 demo benchmark and exit")
    parser.add_argument("--abs2-shape", default="4096,128", help="Complex input shape for abs2 demo, e.g. 4096,128")
    parser.add_argument("--abs2-dtype", default="complex128", choices=["complex64", "complex128"], help="Complex dtype for abs2 demo")
    parser.add_argument(
        "--abs2-backend",
        default="auto",
        choices=["auto", "numpy_abs_square", "numpy_inplace_complex", "numba_vectorize", "conjugate_multiply", "float_view_dot"],
        help="Backend for complex abs2 demo",
    )
    parser.add_argument("--abs2-overwrite-input", action="store_true", help="Allow abs2 demo to overwrite input for lower memory use")
    parser.add_argument("--show-complex-module", action="store_true", help="Show built-in complex() module spec and exit")
    parser.add_argument("--complex-demo", action="store_true", help="Run built-in complex() module demo and exit")
    parser.add_argument("--complex-a", default="1+2j", help="First complex literal for demo, e.g. 1+2j")
    parser.add_argument("--complex-b", default="3+4j", help="Second complex literal for demo, e.g. 3+4j")
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
    if args.resilience_factor is not None:
        config.resilience_factor = args.resilience_factor
    if args.negative_forgetting_rate is not None:
        config.negative_forgetting_rate = args.negative_forgetting_rate

    if args.simulate:
        sim = simulate_future_system(config, steps=max(1, int(args.simulation_steps)))
        print(
            json.dumps(
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
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.simulate_vr:
        sim = simulate_vr_resilience(config, steps=max(1, int(args.vr_steps)))
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "mode": "vr_simulation",
                    "steps": sim.steps,
                    "foresight_score": round(sim.foresight_score, 4),
                    "avoidance_success_rate": round(sim.avoidance_success_rate, 4),
                    "pain_resilience_score": round(sim.pain_resilience_score, 4),
                    "continuity_score": round(sim.continuity_score, 4),
                    "negative_memory_pressure": round(sim.negative_memory_pressure, 4),
                    "recommended_stack": sim.recommended_stack,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.simulate_vr_timetravel:
        sim = simulate_vr_timetravel(config, steps=max(1, int(args.vr_timetravel_steps)))
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "mode": "vr_timetravel_simulation",
                    "steps": sim.steps,
                    "timeline_glimpse_score": round(sim.timeline_glimpse_score, 4),
                    "hazard_avoidance_score": round(sim.hazard_avoidance_score, 4),
                    "frame_stability_score": round(sim.frame_stability_score, 4),
                    "latency_risk": round(sim.latency_risk, 4),
                    "recommended_modules": sim.recommended_modules,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.simulate_vr_autonomy:
        sim = simulate_vr_autonomy_accessibility(config, steps=max(1, int(args.vr_autonomy_steps)))
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "mode": "vr_autonomy_accessibility_simulation",
                    "steps": sim.steps,
                    "automation_reliability": round(sim.automation_reliability, 4),
                    "update_safety_score": round(sim.update_safety_score, 4),
                    "autopilot_world_stability": round(sim.autopilot_world_stability, 4),
                    "haptic_accessibility_score": round(sim.haptic_accessibility_score, 4),
                    "hearing_accessibility_score": round(sim.hearing_accessibility_score, 4),
                    "mental_mode_safety_score": round(sim.mental_mode_safety_score, 4),
                    "overall_safety_score": round(sim.overall_safety_score, 4),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_github_modules:
        catalog = build_github_module_catalog()
        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "repo": m.repo,
                        "url": m.url,
                        "focus": m.focus,
                        "why_it_matters": m.why_it_matters,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_openai_modules:
        catalog = build_openai_module_catalog()
        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "url": m.url,
                        "focus": m.focus,
                        "why_it_matters": m.why_it_matters,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_dasroot_2026_modules:
        catalog = build_dasroot_emerging_python_2026_modules()
        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "category": m.category,
                        "summary": m.summary,
                        "python_stack": m.python_stack,
                        "suggested_application": m.suggested_application,
                        "cited_versions": m.cited_versions,
                        "compliance_notes": m.compliance_notes,
                        "source_url": m.source_url,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_linkedin_python2025_modules:
        catalog = build_linkedin_python_trends_2025_modules()
        print(
            json.dumps(
                [
                    {
                        "trend_id": m.trend_id,
                        "trend_name": m.trend_name,
                        "summary": m.summary,
                        "python_stack": m.python_stack,
                        "suggested_application": m.suggested_application,
                        "learning_path": m.learning_path,
                        "source_url": m.source_url,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_vr_modules:
        catalog = build_vr_module_catalog()
        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "url": m.url,
                        "language_mix": m.language_mix,
                        "focus": m.focus,
                        "why_it_matters": m.why_it_matters,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_vr_timetravel_modules:
        catalog = build_semanticproxy_vr_timetravel_modules()
        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "python_stack": m.python_stack,
                        "mixed_stack": m.mixed_stack,
                        "focus": m.focus,
                        "source_url": m.source_url,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_vr_autonomy_modules:
        catalog = build_vr_autonomy_accessibility_modules()
        print(
            json.dumps(
                [
                    {
                        "module_id": m.module_id,
                        "name": m.name,
                        "category": m.category,
                        "summary": m.summary,
                        "python_stack": m.python_stack,
                        "mixed_stack": m.mixed_stack,
                        "target_outcome": m.target_outcome,
                        "compliance_notes": m.compliance_notes,
                        "sources": m.sources,
                    }
                    for m in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_bestprojectideas_ai50:
        catalog = build_bestprojectideas_ai50_catalog()
        print(
            json.dumps(
                [
                    {
                        "idea_id": item.idea_id,
                        "level": item.level,
                        "title": item.title,
                        "languages": item.languages,
                        "source_url": item.source_url,
                    }
                    for item in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.list_requested_addons32:
        catalog = build_requested_addons32_catalog()
        print(
            json.dumps(
                [
                    {
                        "addon_id": item.addon_id,
                        "title": item.title,
                        "primary_python_stack": item.primary_python_stack,
                        "optional_mixed_languages": item.optional_mixed_languages,
                        "focus": item.focus,
                    }
                    for item in catalog
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.show_nsp_module:
        print(json.dumps(build_nsp_bert_module_spec(), indent=2, ensure_ascii=False))
        return 0

    if args.nsp_predict:
        if not args.sentence_a.strip() or not args.sentence_b.strip():
            print("--nsp-predict requires both --sentence-a and --sentence-b.")
            return 2
        nsp_out = run_nsp_prediction(
            sentence_a=args.sentence_a,
            sentence_b=args.sentence_b,
            model_name=args.nsp_model,
            backend=args.nsp_backend,
        )
        print(
            json.dumps(
                {
                    "module": build_nsp_bert_module_spec()["name"],
                    "last_updated": build_nsp_bert_module_spec()["last_updated"],
                    "backend": nsp_out.backend,
                    "model": nsp_out.model,
                    "is_consecutive": nsp_out.is_consecutive,
                    "confidence": nsp_out.confidence,
                    "sentence_a": nsp_out.sentence_a,
                    "sentence_b": nsp_out.sentence_b,
                    "details": nsp_out.details,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.show_abs2_module:
        catalog = build_complex_abs2_module_catalog()
        print(
            json.dumps(
                {
                    **build_complex_abs2_module_spec(),
                    "methods": [
                        {
                            "name": m.name,
                            "backend_key": m.backend_key,
                            "preserves_input": m.preserves_input,
                            "output_dtype": m.output_dtype,
                            "memory_note": m.memory_note,
                            "source_url": m.source_url,
                        }
                        for m in catalog
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.abs2_demo:
        try:
            import importlib

            np_mod = importlib.import_module("numpy")
            shape = _parse_shape_csv(args.abs2_shape)
            real_dtype = np_mod.float64 if args.abs2_dtype == "complex128" else np_mod.float32
            complex_dtype = np_mod.complex128 if args.abs2_dtype == "complex128" else np_mod.complex64
            rng = np_mod.random.default_rng(42)
            arr = (
                rng.standard_normal(size=shape, dtype=real_dtype)
                + 1j * rng.standard_normal(size=shape, dtype=real_dtype)
            ).astype(complex_dtype, copy=False)
            arr_before = arr.copy()

            out, meta = compute_complex_abs2(
                arr=arr,
                backend=args.abs2_backend,
                preserve_input=not args.abs2_overwrite_input,
            )
            input_preserved = bool(np_mod.array_equal(arr, arr_before))
            preview = np_mod.asarray(out).reshape(-1)[:5].tolist()

            print(
                json.dumps(
                    {
                        "module": build_complex_abs2_module_spec()["name"],
                        "source_url": build_complex_abs2_module_spec()["source_url"],
                        "shape": shape,
                        "dtype": args.abs2_dtype,
                        "backend_requested": args.abs2_backend,
                        "backend_used": meta["backend_used"],
                        "preserve_input_requested": not args.abs2_overwrite_input,
                        "input_preserved": input_preserved,
                        "input_nbytes": meta["input_nbytes"],
                        "output_nbytes": meta["output_nbytes"],
                        "peak_estimate_bytes": meta["peak_estimate_bytes"],
                        "peak_estimate_human": bytes_to_human(meta["peak_estimate_bytes"]),
                        "output_preview": preview,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"abs2 demo failed: {exc}")
            return 2

    if args.show_complex_module:
        mod = build_realpython_complex_module()
        print(
            json.dumps(
                {
                    "name": mod.name,
                    "short_description": mod.short_description,
                    "last_verified": mod.last_verified,
                    "constructor_syntax": mod.constructor_syntax,
                    "arguments": mod.arguments,
                    "return_value": mod.return_value,
                    "examples": mod.examples,
                    "methods": mod.methods,
                    "common_use_cases": mod.common_use_cases,
                    "real_world_example": mod.real_world_example,
                    "related_resources": mod.related_resources,
                    "source_url": mod.source_url,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.complex_demo:
        mod = build_realpython_complex_module()
        try:
            demo = run_complex_module_demo(args.complex_a, args.complex_b)
            print(
                json.dumps(
                    {
                        "module": mod.name,
                        "source_url": mod.source_url,
                        "demo": demo,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"complex demo failed: {exc}")
            return 2

    if not args.goal.strip():
        print(
            "A non-empty --goal is required unless --simulate, --simulate-vr, "
            "--simulate-vr-timetravel, --simulate-vr-autonomy, --list-github-modules, --list-openai-modules, "
            "--list-dasroot-2026-modules, --list-linkedin-python2025-modules, "
            "--list-vr-modules, --list-vr-timetravel-modules, --list-vr-autonomy-modules, "
            "--list-bestprojectideas-ai50, --list-requested-addons32, "
            "--show-nsp-module, --nsp-predict, --show-abs2-module, --abs2-demo, "
            "--show-complex-module, or --complex-demo is used."
        )
        return 2

    if args.chat_completions:
        config.use_responses_api = False
    if args.verbose:
        config.verbose = True

    if not config.api_key:
        print("OPENAI_API_KEY is required for agent run. Use --simulate to run local simulation mode.")
        return 2

    agent = CodexUnifiedAgent(config)
    try:
        result = agent.run(args.goal)
    finally:
        agent.close()

    if args.json:
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "model": config.model,
                    "workspace": str(config.workspace),
                    "turns_used": result.turns_used,
                    "tool_calls": result.tool_calls,
                    "task_summary": result.task_summary,
                    "final_answer": result.final_answer,
                },
                indent=2,
                ensure_ascii=False,
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















