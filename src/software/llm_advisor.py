"""
Optional large-language-model advisor for report wording.

The LLM must never replace the model output or clinician review. It receives a
compact structured case summary and returns narrative suggestions for the final
report. By default it is disabled and the software uses rule-based advice only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMAdvisorConfig:
    enabled: bool = False
    api_base: str = ""
    api_key_env: str = "LLM_API_KEY"
    model: str = ""
    timeout_seconds: int = 60
    temperature: float = 0.2


class LLMAdvisor:
    """OpenAI-compatible chat-completions advisor."""

    def __init__(self, config: Optional[LLMAdvisorConfig] = None):
        self.config = config or LLMAdvisorConfig()

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "LLMAdvisor":
        config = config or {}
        return cls(
            LLMAdvisorConfig(
                enabled=bool(config.get("enabled", False)),
                api_base=str(config.get("api_base", "") or "").rstrip("/"),
                api_key_env=str(config.get("api_key_env", "LLM_API_KEY")),
                model=str(config.get("model", "") or ""),
                timeout_seconds=int(config.get("timeout_seconds", 60)),
                temperature=float(config.get("temperature", 0.2)),
            )
        )

    def available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.api_base or not self.config.model:
            return False
        return bool(os.environ.get(self.config.api_key_env))

    def generate_advice(self, case_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate LLM advice as a JSON object, or return None on failure."""
        if not self.available():
            return None

        endpoint = f"{self.config.api_base}/chat/completions"
        api_key = os.environ[self.config.api_key_env]
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是医学影像AI系统的报告助手。你只能基于输入的结构化AI结果"
                        "生成谨慎的报告建议，不能声称确诊，不能替代医生。"
                        "输出必须是JSON，字段包括 impression, risk_notes, next_steps, disclaimer。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(case_summary, ensure_ascii=False, indent=2),
                },
            ],
        }

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            raw = json.loads(body)
            content = raw["choices"][0]["message"]["content"]
            return self._parse_json_content(content)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
            return None

    @staticmethod
    def _parse_json_content(content: str) -> Optional[Dict[str, Any]]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None
        return parsed
