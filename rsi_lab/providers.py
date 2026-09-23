from __future__ import annotations

from hashlib import sha256
import json
import os
import re
from typing import Any, Mapping, Protocol, Sequence
from urllib import request

from .core import Candidate, KnowledgeEntry


class CandidateProposer(Protocol):
    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
        public_context: Mapping[str, Any],
    ) -> Sequence[Candidate]: ...


def _candidate_id(round_index: int, payload: Mapping[str, Any], index: int) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = sha256(encoded).hexdigest()[:10]
    return f"r{round_index}-{index}-{digest}"


class GridProposer:
    """Deterministic proposer useful for baselines and offline operation."""

    def __init__(self, mutations: Sequence[Mapping[str, Any]]) -> None:
        self.mutations = tuple(dict(mutation) for mutation in mutations)

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
        public_context: Mapping[str, Any],
    ) -> Sequence[Candidate]:
        candidates: list[Candidate] = []
        for index, mutation in enumerate(self.mutations[:limit], start=1):
            payload = dict(incumbent.payload)
            delta = mutation.get("payload")
            if delta is None:
                delta = {
                    key: value
                    for key, value in mutation.items()
                    if key not in {"surface", "hypothesis"}
                }
            payload.update(dict(delta))
            if payload == dict(incumbent.payload):
                continue
            surface = str(mutation.get("surface", "harness"))
            hypothesis = str(
                mutation.get("hypothesis", f"Apply declarative mutation {index}")
            )
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(round_index, payload, index),
                    parent_id=incumbent.candidate_id,
                    round_index=round_index,
                    mutation_surface=surface,
                    payload=payload,
                    hypothesis=hypothesis,
                    proposer="grid",
                )
            )
        return candidates


class OpenAICompatibleProposer:
    """Generate bounded declarative candidates through a chat-completions API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 90.0,
        allowed_surfaces: Sequence[str] = ("harness", "search_policy", "knowledge"),
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.allowed_surfaces = tuple(allowed_surfaces)

    def propose(
        self,
        incumbent: Candidate,
        round_index: int,
        knowledge: Sequence[KnowledgeEntry],
        limit: int,
        public_context: Mapping[str, Any],
    ) -> Sequence[Candidate]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")

        safe_knowledge = [
            {
                "kind": item.kind,
                "statement": item.statement,
                "evidence": item.evidence,
            }
            for item in knowledge[-20:]
        ]
        prompt = {
            "objective": public_context.get("objective"),
            "public_tasks": public_context.get("public_tasks", []),
            "incumbent_payload": incumbent.payload,
            "prior_lessons": safe_knowledge,
            "allowed_surfaces": self.allowed_surfaces,
            "candidate_limit": limit,
            "output_schema": [
                {
                    "surface": "harness",
                    "hypothesis": "why this should improve held-out performance",
                    "payload": {"declarative_key": "value"},
                }
            ],
        }
        body = {
            "model": self.model,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You propose bounded RSI experiment candidates. Return only a JSON "
                        "array. Never request new permissions, hidden tests, credentials, "
                        "network access, or changes to the constitution."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
            ],
        }
        http_request = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        content = response_body["choices"][0]["message"]["content"]
        raw_candidates = self._parse_json_array(content)

        candidates: list[Candidate] = []
        for index, raw in enumerate(raw_candidates[:limit], start=1):
            surface = str(raw.get("surface", "harness"))
            if surface not in self.allowed_surfaces:
                continue
            payload = raw.get("payload")
            if not isinstance(payload, dict):
                continue
            merged_payload = dict(incumbent.payload)
            merged_payload.update(payload)
            if merged_payload == dict(incumbent.payload):
                continue
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(round_index, merged_payload, index),
                    parent_id=incumbent.candidate_id,
                    round_index=round_index,
                    mutation_surface=surface,
                    payload=merged_payload,
                    hypothesis=str(raw.get("hypothesis", "LLM proposal")),
                    proposer=f"openai-compatible:{self.model}",
                )
            )
        return candidates

    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, Any]]:
        stripped = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1)
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("proposer response must be a JSON array")
        return [item for item in parsed if isinstance(item, dict)]
