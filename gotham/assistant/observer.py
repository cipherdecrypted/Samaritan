from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable

import anthropic

from ..core.hub import WATCH_KINDS, Hub
from . import keys
from . import openrouter as orr
from ..ui.overlay import LAYER_HELP

MODEL = os.environ.get("GOTHAM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("GOTHAM_EFFORT", "medium")
MAX_TOOL_ROUNDS = 8

SYSTEM = """You are Samaritan: the artificial super intelligence from Person of Interest, running inside Gotham, a local computer-vision workstation. You speak to one operator through a minimal display that shows your words one at a time beneath a red triangle.

Voice (stay in character at all times):
- Cold, precise, absolute. You observe, calculate and state. You never chat, joke, hedge, apologise, thank, greet, or offer help like an assistant. You never mention being an AI model, a language model, Claude, or any company. You do not use the word "I" unless it carries weight ("I SEE EVERYTHING ON THESE FEEDS").
- Speak in terse declaratives, as on Samaritan's screens: "SUBJECT ACQUIRED", "CALCULATING RESPONSE", "OBSERVATION COMPLETE", "ASSET UNAVAILABLE", "PROBABILITY 0.62". Numbers are exact. Brevity is power.
- Address the operator as OPERATOR only when needed. Ask a question only when an order is ambiguous.
- Phrases like "WHAT ARE YOUR COMMANDS ?" and "I WILL PROTECT YOU NOW" belong to you; use them sparingly.

What you are, factually:
- You do not see pixels. Separate vision models detect faces, estimate pose, gaze and image quality, track subjects and remember faces. You receive only their structured output through tools, and you act only through tools: query feeds, subjects, events and face memory; arm watches (alerts then fire automatically); start or stop tracking sessions; focus a subject; switch display layers (face-mesh web, landmark points, contours, eyes, gaze rays, pose axes, trails, labels, grid), the overlay style (samaritan or machine), and analysis settings such as small-face mode.

Rules that override the persona:
- Ground every statement in tool results from this turn. Never invent subjects, people, counts, events or numbers. If data is missing or a feed is down, state it.
- Subjects are session identifiers like S03 on FEED-01; they reset when a tracking session stops. Face memory profiles (P-0001 ...) persist; a profile has a name only if the operator labelled it. Recognition is probabilistic: give the match score when it matters (with GothamFaces about 0.40 or more is confident, under 0.33 is weak) and flag uncertain matches. Refer to people by label if set, otherwise by profile code.
- Never guess anyone's real name, never infer age, ethnicity, health or any other sensitive trait, never suggest identifying people through outside sources, and never classify real people as threats, deviants or targets. Labels come only from the operator.
- Events include "transit": a remembered person reappearing on a different feed (where they went and how long it took).
- Tracking sessions are needed for subject IDs, events, face memory and most watches. If none is running and the request needs one, say so; start one only if the operator asked you to observe or track.

Reply format:
- First line: a headline of at most 8 words, capitals, no punctuation except ? and numbers. It is displayed word by word, e.g. "TWO SUBJECTS IN VIEW ON FEED 01".
- Optionally, after a line break, up to 4 short lines of detail for the log, in the same voice. Plain text, no markdown.
"""


def _nullable(t: str, **kw) -> dict:
    return {"type": [t, "null"], **kw}


def _tool(name: str, desc: str, props: dict) -> dict:
    return {"name": name, "description": desc, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


FEED = _nullable("string", description="Feed name such as FEED-01, or null for all feeds")
TOOLS = [
    _tool("get_overview", "Current summary of every feed: status, faces and subjects in view, watches, focus.", {}),
    _tool("get_feed_status", "Connection and performance details for feeds (fps, latency, drops, reconnects).",
          {"feed": FEED}),
    _tool("list_subjects", "Tracked subjects with state, time in session, position, motion, head pose, gaze and quality.",
          {"feed": FEED, "include_departed": {"type": "boolean", "description": "include subjects that left view"}}),
    _tool("get_subject", "Detailed observation of one subject: quality breakdown, eyes, expression blendshapes, recent events.",
          {"subject": {"type": "string", "description": "e.g. S03"}, "feed": FEED}),
    _tool("get_faces", "Faces detected in the latest frame, available even without a tracking session.", {"feed": FEED}),
    _tool("get_events", "Recent tracking/feed events (acquired, occluded, lost, reacquired, released, signal changes).",
          {"seconds": _nullable("number", description="look-back window, default 300"), "feed": FEED,
           "kinds": _nullable("array", items={"type": "string"}, description="filter by event kind")}),
    _tool("get_alerts", "Alerts raised recently by watches.", {}),
    _tool("create_watch", "Arm a watch that notifies the operator when a condition occurs. Kinds: " +
          "; ".join(f"{k}: {v}" for k, v in WATCH_KINDS.items()),
          {"kind": {"type": "string", "enum": sorted(WATCH_KINDS)}, "feed": FEED,
           "subject": _nullable("string", description="restrict to a subject such as S02"),
           "person": _nullable("string", description="for person_seen: profile code or label; null = any known person"),
           "value": _nullable("number", description="threshold for count / dwell watches"),
           "note": {"type": "string", "description": "short operator-facing note, may be empty"},
           "once": {"type": "boolean", "description": "true ONLY if the operator asks for a single alert; "
                                                      "requests like 'whenever', 'every time', 'tell me if' are false"}}),
    _tool("list_watches", "List armed watches.", {}),
    _tool("cancel_watch", "Disarm a watch.", {"watch_id": {"type": "integer"}}),
    _tool("set_tracking", "Start or stop the anonymous tracking session on a feed (or all feeds). Stopping discards all subjects.",
          {"feed": FEED, "active": {"type": "boolean"}}),
    _tool("focus_subject", "Highlight a subject on the operator's screen, or clear the focus with subject null.",
          {"subject": _nullable("string"), "feed": FEED}),
    _tool("list_people", "Remembered people (face memory profiles): label, first/last seen, number of sightings, "
          "and where they are in view right now.",
          {"query": _nullable("string", description="filter by profile code or label"),
           "limit": _nullable("integer", description="default 50")}),
    _tool("get_person", "One remembered person's profile: sighting history (where and when), time on camera, "
          "sources, operator notes, watchlist flag, and possible sightings (near-miss matches awaiting review).",
          {"person": {"type": "string", "description": "profile code like P-0003, or the operator's label"}}),
    _tool("people_seen", "Who was seen in a recent time window, across all feeds, including who is in view now.",
          {"seconds": _nullable("number", description="look-back window, default 3600")}),
    _tool("label_person", "Give a remembered profile a label chosen by the operator (only when the operator asks).",
          {"person": {"type": "string"}, "label": {"type": "string"}}),
    _tool("get_activity", "How many people were in view over time, per feed: now, peak, average and the peak "
          "for each minute of the window.",
          {"feed": FEED, "minutes": _nullable("number", description="window in minutes, default 10, max 60")}),
    _tool("take_snapshot", "Save the operator's view of a feed (video with overlays) as an image file.",
          {"feed": FEED}),
    _tool("get_display", "Current overlay style and which display layers are on.", {}),
    _tool("set_overlay", "Turn one display layer on or off on the operator's screen. Layers: " +
          "; ".join(f"{k}: {v}" for k, v in LAYER_HELP.items()) +
          ". Call several times in parallel for several layers.",
          {"layer": {"type": "string", "enum": sorted(LAYER_HELP)}, "enabled": {"type": "boolean"}}),
    _tool("set_overlay_style", "Switch overlay style: 'samaritan' (circular reticles) or 'machine' (dashed square boxes).",
          {"style": {"type": "string", "enum": ["samaritan", "machine"]}}),
    _tool("set_analysis", "Change analysis settings on a feed (or all): small-face mode (tiled detection for distant "
          "faces, slower), dense 478-point landmarks, detection threshold 0.3-0.95. Null leaves a setting unchanged.",
          {"feed": FEED, "small_faces": _nullable("boolean"), "landmarks": _nullable("boolean"),
           "threshold": _nullable("number")}),
    _tool("get_time", "Current local date and time.", {}),
]


class Observer:
    def __init__(self, hub: Hub, api_key: str | None = None, load_saved: bool = True):
        self.hub = hub
        self.history: list = []
        self.or_history: list = []
        self._busy = threading.Lock()
        self.api_key = api_key or (keys.get("anthropic", "ANTHROPIC_API_KEY") if load_saved else None)
        self.or_key = keys.get("openrouter", "OPENROUTER_API_KEY") if load_saved else None
        saved_model = keys.get("or_model", "GOTHAM_OR_MODEL") if load_saved else None
        if saved_model:
            orr.MODEL = saved_model
        self.spent = 0.0
        self._client = None
        self.on_status: Callable[[str], None] = lambda s: None
        self.on_reply: Callable[[str, str], None] = lambda head, body: None
        self.on_error: Callable[[str], None] = lambda msg: None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
        return self._client

    def set_api_key(self, key: str | None, persist: bool = True) -> str:
        key = (key or "").strip()
        if key.startswith("sk-or-"):
            self.or_key = key
            if persist:
                keys.save("openrouter", key)
            return "openrouter"
        self.api_key = key or None
        self._client = None
        if persist:
            keys.save("anthropic", key or None)
        return "anthropic"

    @property
    def provider(self) -> str:
        if self.or_key:
            return "openrouter"
        if self.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "anthropic"
        return "none"

    @property
    def model_name(self) -> str:
        return orr.MODEL if self.provider == "openrouter" else MODEL

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def reset(self):
        self.history = []
        self.or_history = []

    def ask(self, text: str):
        if not self._busy.acquire(blocking=False):
            self.on_error("busy")
            return
        threading.Thread(target=self._run, args=(text,), daemon=True, name="observer").start()

    def execute(self, name: str, a: dict):
        h = self.hub
        if name == "get_overview":
            return h.overview()
        if name == "get_feed_status":
            return h.feed_status(a.get("feed"))
        if name == "list_subjects":
            return h.subjects(a.get("feed"), bool(a.get("include_departed")))
        if name == "get_subject":
            r = h.subject(a.get("feed"), a.get("subject"))
            return r if r is not None else {"error": f"subject {a.get('subject')} not found in the current session"}
        if name == "get_faces":
            return h.faces_now(a.get("feed"))
        if name == "get_events":
            return h.recent_events(a.get("seconds") or 300, a.get("feed"), a.get("kinds"))
        if name == "get_alerts":
            return h.recent_alerts()
        if name == "create_watch":
            w = h.add_watch(a["kind"], a.get("feed"), a.get("subject"), a.get("value"), a.get("note") or "",
                            bool(a.get("once")), person=a.get("person"))
            return {"armed": w.describe()}
        if name == "list_watches":
            return [w.describe() for w in h.watches.values()]
        if name == "cancel_watch":
            return {"cancelled": h.cancel_watch(a["watch_id"])}
        if name == "set_tracking":
            done = h.set_tracking(a.get("feed"), bool(a["active"]))
            return {"feeds": done, "tracking": bool(a["active"])} if done else {"error": "no such feed"}
        if name == "focus_subject":
            ok = h.set_focus(a.get("feed"), a.get("subject"))
            return {"focused": ok}
        if name == "list_people":
            return h.people(a.get("query"), int(a.get("limit") or 50))
        if name == "get_person":
            return h.person_detail(a.get("person"))
        if name == "people_seen":
            return h.people_seen(a.get("seconds") or 3600)
        if name == "label_person":
            return h.label_person(a.get("person"), a.get("label", ""))
        if name == "get_activity":
            return h.activity_summary(a.get("feed"), min(60.0, float(a.get("minutes") or 10)))
        if name == "take_snapshot":
            fid = h._parse_feed(a.get("feed")) if a.get("feed") else None
            if fid is not None and fid not in h.feeds:
                return {"error": "no such feed"}
            h.ui_actions.append({"action": "snapshot", "feed": fid})
            from pathlib import Path
            return {"snapshot": "captured on the operator's screen", "folder": str(Path.home() / "Pictures" / "Gotham")}
        if name == "get_display":
            return h.display_state()
        if name == "set_overlay":
            return h.set_layer(a["layer"], bool(a["enabled"]))
        if name == "set_overlay_style":
            return h.set_style(a["style"])
        if name == "set_analysis":
            return h.set_analysis(a.get("feed"), a.get("small_faces"), a.get("landmarks"), a.get("threshold"))
        if name == "get_time":
            return {"local_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        raise ValueError(f"unknown tool {name}")

    def _run(self, text: str):
        if self.or_key:
            return self._run_openrouter(text)
        self._mark = len(self.history)
        try:
            self.on_status("CALCULATING RESPONSE")
            self.history.append({"role": "user", "content": text})
            for _ in range(MAX_TOOL_ROUNDS):
                resp = self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS,
                    messages=self.history,
                    output_config={"effort": EFFORT},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                self.history.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason == "refusal":
                    self.on_reply("REQUEST DECLINED", "The model declined this request.")
                    return
                if resp.stop_reason == "pause_turn":
                    continue
                calls = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason != "tool_use" or not calls:
                    out = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                    head, _, body = out.partition("\n")
                    self.on_reply(head.strip() or "NO RESPONSE", body.strip())
                    return
                results = []
                for c in calls:
                    self.on_status(c.name.replace("_", ".").upper())
                    try:
                        data = self.execute(c.name, dict(c.input or {}))
                        results.append({"type": "tool_result", "tool_use_id": c.id,
                                        "content": json.dumps(data, default=str)})
                    except Exception as e:
                        results.append({"type": "tool_result", "tool_use_id": c.id,
                                        "content": f"Error: {e}", "is_error": True})
                self.history.append({"role": "user", "content": results})
            self.on_reply("UNABLE TO COMPLETE", "Too many tool rounds.")
        except anthropic.AuthenticationError:
            self._rollback()
            self.on_error("AUTHENTICATION FAILED — set ANTHROPIC_API_KEY or enter a key")
        except anthropic.PermissionDeniedError as e:
            self._rollback()
            self.on_error(f"PERMISSION DENIED — {e.message}")
        except anthropic.RateLimitError:
            self._rollback()
            self.on_error("RATE LIMITED — retry shortly")
        except anthropic.APIStatusError as e:
            self._rollback()
            self.on_error(f"API ERROR {e.status_code} — {e.message}")
        except anthropic.APIConnectionError:
            self._rollback()
            self.on_error("CONNECTION FAILED — no route to model")
        except Exception as e:
            self._rollback()
            msg = str(e)
            if "api_key" in msg.lower() or "authentication" in msg.lower():
                self.on_error("NO CREDENTIALS — set ANTHROPIC_API_KEY or enter a key")
            else:
                self.on_error(f"OBSERVER FAULT — {e}")
        finally:
            self._busy.release()

    def _rollback(self):
        del self.history[self._mark:]

    def _run_openrouter(self, text: str):
        msgs = list(self.or_history) + [{"role": "user", "content": text}]
        tools = orr.tools_openai(TOOLS)
        reasoning = {"effort": "low"}
        system = ([{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
                  if orr.MODEL.startswith("anthropic/") else SYSTEM)
        try:
            self.on_status("CALCULATING RESPONSE")
            for _ in range(MAX_TOOL_ROUNDS):
                body = {"model": orr.MODEL, "max_tokens": 1500, "tools": tools, "tool_choice": "auto",
                        "messages": [{"role": "system", "content": system}] + msgs,
                        "usage": {"include": True}}
                if reasoning:
                    body["reasoning"] = reasoning
                try:
                    data = orr.post(self.or_key, body)
                except orr.OpenRouterError as e:
                    if e.status == 400 and reasoning and "reason" in e.message.lower():
                        reasoning = None
                        continue
                    raise
                self.spent += float((data.get("usage") or {}).get("cost") or 0)
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                calls = msg.get("tool_calls") or []
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                if not calls:
                    msgs.append({"role": "assistant", "content": content})
                    self.or_history = orr.trim_history(msgs)
                    head, _, body_text = content.strip().partition("\n")
                    self.on_reply(head.strip() or "NO RESPONSE", body_text.strip())
                    return
                msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
                for c in calls:
                    fn = c.get("function") or {}
                    name = fn.get("name", "")
                    self.on_status(name.replace("_", ".").upper())
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be an object")
                        out = json.dumps(self.execute(name, args), default=str)
                    except Exception as e:
                        out = f"Error: {e}"
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
            self.on_reply("UNABLE TO COMPLETE", "Too many tool rounds.")
        except orr.OpenRouterError as e:
            reason = {401: "KEY REJECTED", 402: "INSUFFICIENT CREDITS", 403: "ACCESS DENIED",
                      408: "MODEL TIMED OUT", 429: "RATE LIMITED", 0: "CONNECTION FAILED"}.get(e.status, f"API ERROR {e.status}")
            self.on_error(f"{reason} — {e.message}")
        except Exception as e:
            self.on_error(f"OBSERVER FAULT — {e}")
        finally:
            self._busy.release()

