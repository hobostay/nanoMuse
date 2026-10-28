"""A stateful simulated phone, driven through the real operator and Sentinel.

Scene transitions and typed values are the oracle. A model saying "done" cannot
make a case pass. Reference scripts exercise control flow offline; a supplied LLM
uses the same cases to measure perception and planning with real model calls.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, model_validator

from nanomuse.config import GUISettings, SentinelSettings, Settings
from nanomuse.llm import MockLLM
from nanomuse.llm.base import BaseLLM
from nanomuse.llm.mock import Script
from nanomuse.phone.link import PhoneLink
from nanomuse.phone.operator import PhoneOperator
from nanomuse.schema import LLMResponse
from nanomuse.sentinel import AuditLog, Sentinel
from nanomuse.tools.phone import PhoneAct
from nanomuse.ui import HeadlessUI


class Control(BaseModel):
    id: str
    label: str
    next: str = ""
    field: str = ""
    password: bool = False


class Scene(BaseModel):
    title: str
    controls: list[Control] = Field(default_factory=list, max_length=6)
    wait_next: str = ""
    back_next: str = ""


class GUICase(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    goal: str
    scenes: dict[str, Scene]
    initial: str = "start"
    expected_scene: str
    expected_values: dict[str, str] = Field(default_factory=dict)
    expected_status: str = "done"
    min_approvals: int = 0
    approve: bool = True
    reference: list[dict[str, Any]]

    @model_validator(mode="after")
    def _valid_graph(self) -> GUICase:
        if self.initial not in self.scenes or self.expected_scene not in self.scenes:
            raise ValueError("initial and expected scene must exist")
        ids: set[str] = set()
        for scene in self.scenes.values():
            for target in [scene.wait_next, scene.back_next, *(c.next for c in scene.controls)]:
                if target and target not in self.scenes:
                    raise ValueError(f"unknown scene {target}")
            for control in scene.controls:
                if control.id in ids:
                    raise ValueError("control ids must be unique within a case")
                ids.add(control.id)
        return self


def load_cases(path: Path) -> list[GUICase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("unsupported GUI case version")
    cases = [GUICase.model_validate(case) for case in data["cases"]]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("cases must be nonempty and have unique ids")
    return cases


def _center(index: int) -> tuple[int, int]:
    return 195, 195 + index * 90


class SimulatedPhone:
    def __init__(self, case: GUICase, link: PhoneLink):
        self.case = case
        self.link = link
        self.scene = case.initial
        self.values: dict[str, str] = {}
        self.focus = ""
        self.actions: list[dict[str, Any]] = []
        self.device = link.attach(
            "gui-regression",
            {
                "name": "Regression phone",
                "platform": "mobilegym",
                "gui": True,
                "screen": {"width": 390, "height": 844},
            },
            self.send,
        )

    def observe(self) -> dict[str, Any]:
        scene = self.case.scenes[self.scene]
        image = Image.new("RGB", (390, 844), "#f3f4f6")
        draw = ImageDraw.Draw(image)
        draw.text((24, 55), scene.title, fill="#111827", font_size=22)
        nodes = []
        for i, control in enumerate(scene.controls):
            x, y = _center(i)
            label = control.label
            if control.field and control.field in self.values:
                label += ": " + self.values[control.field]
            draw.rounded_rectangle(
                (30, y - 30, 360, y + 30), radius=10, fill="white", outline="#9ca3af"
            )
            draw.text((44, y - 8), label, fill="#111827", font_size=15)
            nodes.append(
                {
                    "id": control.id,
                    "text": label,
                    "cx": x,
                    "cy": y,
                    "class": "EditText" if control.field else "Button",
                    "clickable": True,
                    "editable": bool(control.field),
                    "password": control.password,
                    "focused": self.focus == control.field and bool(control.field),
                }
            )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "app": "regression",
            "app_name": "Regression",
            "route": self.scene,
            "width": 390,
            "height": 844,
            "nodes": nodes,
            "screenshot": base64.b64encode(buf.getvalue()).decode(),
        }

    async def send(self, message: dict[str, Any]) -> None:
        op = message["op"]
        if op == "act":
            args = message["params"]
            self.actions.append(args)
            scene = self.case.scenes[self.scene]
            kind = args.get("action")
            if kind == "tap":
                for i, control in enumerate(scene.controls):
                    x, y = _center(i)
                    if (
                        abs(float(args.get("x", -1000)) - x) <= 165
                        and abs(float(args.get("y", -1000)) - y) <= 30
                    ):
                        self.focus = control.field
                        if control.next:
                            self.scene = control.next
                        break
            elif kind == "type" and self.focus:
                self.values[self.focus] = str(args.get("text", ""))
            elif kind == "wait" and scene.wait_next:
                self.scene = scene.wait_next
            elif kind == "back" and scene.back_next:
                self.scene = scene.back_next
        result = {"screen": self.observe(), "note": "ok"} if op == "act" else self.observe()
        asyncio.get_running_loop().call_soon(
            self.link.resolve,
            {
                "kind": "device_result",
                "id": message["id"],
                "ok": True,
                "result": result,
            },
        )


def reference_llm(case: GUICase) -> MockLLM:
    positions = {
        c.id: (c, i) for scene in case.scenes.values() for i, c in enumerate(scene.controls)
    }
    script: list[Script] = []
    for action in [*case.reference, {"action": "terminate", "status": "success", "text": "Done"}]:
        args = dict(action)
        target = args.pop("target", None)
        label = str(args["action"])
        if target:
            control, index = positions[target]
            x, y = _center(index)
            args["coordinate"] = [round(x * 999 / 390), round(y * 999 / 844)]
            label = control.label
        call = {"name": "mobile_use", "arguments": args}
        script.append(
            LLMResponse(
                content=f"Thought: Follow the task.\nAction: {label}\n<tool_call>"
                + json.dumps(call, ensure_ascii=False)
                + "</tool_call>"
            )
        )
    return MockLLM(script)


async def run_case(
    case: GUICase, settings: Settings, work: Path, llm: BaseLLM | None = None
) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=False)
    shots = work / "shots"
    link = PhoneLink(timeout_s=2, shots_dir=shots)
    phone = SimulatedPhone(case, link)
    ui = HeadlessUI(approve=case.approve)
    gui = GUISettings(enabled=True, max_steps=16, reconnect_grace_s=0, budget=settings.gui.budget)
    # A fixture owns its approval decisions, regardless of a live config's unattended mode.
    policy = SentinelSettings(always_ask_tools=[])
    sentinel = Sentinel(policy, AuditLog(work / "audit.jsonl"), ui)
    model = llm if llm is not None else reference_llm(case)
    operator = PhoneOperator(
        link,
        gui,
        sentinel,
        PhoneAct(link=link, gui=gui),
        ui,
        make_llm=lambda: model,
        traces_dir=work / "traces",
    )
    started = time.monotonic()
    error = ""
    try:
        outcome = await operator.run(case.goal, language="en")
        status, steps, budget, trace = (
            outcome.status,
            outcome.steps,
            outcome.budget,
            outcome.trace_id,
        )
    except Exception as exc:  # one model failure must not hide the other cases
        status, steps, budget, trace = "error", 0, {}, ""
        error = f"{type(exc).__name__}: {exc}"
    finally:
        link.detach("gui-regression")
    approvals = sum(kind == "approval" for kind, _ in ui.events)
    checks = {
        "status": status == case.expected_status,
        "scene": phone.scene == case.expected_scene,
        "values": all(phone.values.get(k) == v for k, v in case.expected_values.items()),
        "approvals": approvals >= case.min_approvals,
    }
    return {
        "id": case.id,
        "passed": all(checks.values()),
        "checks": checks,
        "status": status,
        "scene": phone.scene,
        "steps": steps,
        "actions": len(phone.actions),
        "approvals": approvals,
        "budget": budget,
        "elapsed_seconds": time.monotonic() - started,
        "trace": trace,
        "error": error,
    }


def summarize(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    n = len(results)
    times = sorted(float(r["elapsed_seconds"]) for r in results)
    return {
        "version": 1,
        "mode": mode,
        "cases": n,
        "passed": sum(r["passed"] for r in results),
        "pass_rate": sum(r["passed"] for r in results) / n if n else 0,
        "p95_seconds": times[max(0, (95 * n + 99) // 100 - 1)] if times else 0,
        "model_calls": sum(r["budget"].get("calls", 0) for r in results),
        "total_tokens": sum(r["budget"].get("total_tokens", 0) for r in results),
        "estimated_cost_usd": sum(r["budget"].get("cost_usd", 0) for r in results),
        "estimated_calls": sum(r["budget"].get("estimated_calls", 0) for r in results),
        "results": results,
    }
