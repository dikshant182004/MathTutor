from __future__ import annotations
import json, os, time, math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from backend.v2.knowledge_graph import prerequisite_edges

try:
    import redis
except Exception:
    redis = None

@dataclass
class SkillState:
    skill: str
    mastery: float = 0.5
    attempts: int = 0
    correct: int = 0
    hints: int = 0
    last_seen: float = 0.0
    recent_errors: list[str] = field(default_factory=list)

class StudentMemoryStore:
    def __init__(self, path: str = ".data/v2_student_memory.json"):
        self.redis_url = os.getenv("REDIS_URL", "")
        self._redis = None
        self._lock = RLock()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if redis and self.redis_url:
            try:
                c = redis.from_url(self.redis_url, decode_responses=True)
                c.ping()
                self._redis = c
            except Exception:
                pass

    def _key(self, student_id: str, suffix: str) -> str:
        if not student_id or ":" in student_id:
            raise ValueError("invalid student_id")
        return f"v2:student:{student_id}:{suffix}"

    def _load(self):
        if not self.path.exists(): return {}
        try: return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError): return {}

    def _save(self, data):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def remember(self, student_id: str, kind: str, key: str, value: str, confidence: float = 0.7):
        if kind not in {"semantic", "procedural", "profile"}:
            raise ValueError("kind must be semantic, procedural or profile")
        payload = {"key": key[:160], "value": value[:1000], "confidence": max(0.0, min(1.0, confidence)), "updated_at": time.time()}
        if self._redis:
            self._redis.hset(self._key(student_id, kind), key[:160], json.dumps(payload))
        else:
            with self._lock:
                data=self._load(); data.setdefault(student_id, {}).setdefault(kind, {})[key[:160]]=payload; self._save(data)
        return payload

    def recall(self, student_id: str, kind: str, limit: int = 20):
        if kind not in {"semantic", "procedural", "profile"}:
            raise ValueError("invalid memory kind")
        if self._redis:
            values=self._redis.hgetall(self._key(student_id, kind)).values()
            items=[json.loads(v) for v in values]
        else:
            items=list(self._load().get(student_id, {}).get(kind, {}).values())
        return sorted(items, key=lambda x:x.get("confidence",0), reverse=True)[:max(1,min(limit,100))]

    def get_skill(self, student_id: str, skill: str) -> SkillState:
        if self._redis:
            raw = self._redis.hget(self._key(student_id, "skills"), skill)
            return SkillState(**json.loads(raw)) if raw else SkillState(skill=skill)
        with self._lock:
            raw = self._load().get(student_id, {}).get("skills", {}).get(skill)
            return SkillState(**raw) if raw else SkillState(skill=skill)

    def _put_skill(self, student_id: str, state: SkillState):
        if self._redis:
            self._redis.hset(self._key(student_id, "skills"), state.skill, json.dumps(asdict(state)))
            return
        with self._lock:
            data = self._load()
            data.setdefault(student_id, {}).setdefault("skills", {})[state.skill] = asdict(state)
            self._save(data)

    @staticmethod
    def _update(mastery, correct, difficulty, hints, response_ms):
        d = max(0.0, min(1.0, difficulty))
        penalty = min(0.2, max(0, hints) * 0.03)
        speed = 0.04 if response_ms is not None and response_ms <= 20000 else -0.04 if response_ms and response_ms >= 180000 else 0
        target = (0.88 if correct else 0.08) - penalty + speed
        weight = 0.12 + 0.18 * d
        return max(0.0, min(1.0, mastery + weight * (target - mastery)))

    def record_attempt(self, student_id, skill, *, correct, difficulty=0.5, hints=0, response_ms=None, error=None):
        skill = (skill or "unknown").strip()
        state = self.get_skill(student_id, skill)
        state.attempts += 1
        state.correct += int(correct)
        state.hints += max(0, hints)
        state.last_seen = time.time()
        state.mastery = self._update(state.mastery, correct, difficulty, hints, response_ms)
        if not correct and error:
            state.recent_errors = (state.recent_errors + [error[:240]])[-10:]
        self._put_skill(student_id, state)
        self._event(student_id, skill, correct, difficulty, hints, response_ms, error)
        return state

    def _event(self, student_id, skill, correct, difficulty, hints, response_ms, error):
        event = {"id": str(int(time.time()*1000)), "student_id": student_id, "skill": skill,
                 "correct": bool(correct), "difficulty": difficulty, "hints": hints,
                 "response_ms": response_ms, "error": error[:240] if error else None, "timestamp": time.time()}
        if self._redis:
            key = self._key(student_id, "events")
            self._redis.lpush(key, json.dumps(event)); self._redis.ltrim(key, 0, 199)
        else:
            with self._lock:
                data = self._load()
                events = data.setdefault(student_id, {}).setdefault("events", [])
                events.insert(0, event); del events[200:]; self._save(data)
        self._graph_event(student_id, event)

    def _graph_event(self, student_id, event):
        graph = self.graph(student_id)
        nid = "skill:" + event["skill"]
        state = self.get_skill(student_id, event["skill"])
        graph["nodes"][nid] = {"id": nid, "type": "skill", "label": state.skill,
                               "mastery": round(state.mastery, 4), "attempts": state.attempts,
                               "correct": state.correct, "status": "mastered" if state.mastery >= .8 else "developing" if state.mastery >= .5 else "weak"}
        if event["error"]:
            mid = "mistake:" + _slug(event["error"])
            graph["nodes"][mid] = {"id": mid, "type": "mistake", "label": event["error"][:80],
                                   "count": graph["nodes"].get(mid, {}).get("count", 0) + 1}
            edge = {"source": mid, "target": nid, "type": "causes"}
            if edge not in graph["edges"]: graph["edges"].append(edge)
        self._save_graph(student_id, graph)

    def graph(self, student_id):
        if self._redis:
            raw = self._redis.get(self._key(student_id, "graph"))
            graph = json.loads(raw) if raw else {"nodes": {}, "edges": []}
        else:
            graph = self._load().get(student_id, {}).get("graph", {"nodes": {}, "edges": []})
        for node in list(graph.get("nodes", {}).values()):
            if node.get("type") == "skill":
                for edge in prerequisite_edges(node["label"]):
                    if edge["source"] not in graph["nodes"]:
                        graph["nodes"][edge["source"]] = {
                            "id": edge["source"], "type": "skill", "label": edge["source"].split(":",1)[1],
                            "mastery": 0.5, "attempts": 0, "correct": 0, "status": "unseen"
                        }
                    if edge not in graph["edges"]:
                        graph["edges"].append(edge)
        return graph

    def _save_graph(self, student_id, graph):
        if self._redis:
            self._redis.set(self._key(student_id, "graph"), json.dumps(graph)); return
        with self._lock:
            data = self._load(); data.setdefault(student_id, {})["graph"] = graph; self._save(data)

    @staticmethod
    def _effective_mastery(skill):
        half_life_days = 45.0
        if not skill.get("last_seen"):
            return skill["mastery"]
        age_days = max(0.0, (time.time() - skill["last_seen"]) / 86400.0)
        retention = math.pow(0.5, age_days / half_life_days)
        return max(0.0, min(1.0, skill["mastery"] * retention))

    def snapshot(self, student_id):
        if self._redis:
            skills = [json.loads(v) for v in self._redis.hgetall(self._key(student_id, "skills")).values()]
            events = [json.loads(v) for v in self._redis.lrange(self._key(student_id, "events"), 0, 19)]
        else:
            doc = self._load().get(student_id, {})
            skills, events = list(doc.get("skills", {}).values()), list(doc.get("events", []))[:20]
        for skill in skills:
            skill["retention"] = round(self._effective_mastery(skill), 4)
            skill["effective_mastery"] = skill["retention"]
        skills.sort(key=lambda x: x["effective_mastery"])
        return {"student_id": student_id, "skills": skills, "weakest": skills[:5], "events": events,
                "semantic_memory": self.recall(student_id, "semantic", 10),
                "procedural_memory": self.recall(student_id, "procedural", 10),
                "profile_memory": self.recall(student_id, "profile", 10),
                "graph": self.graph(student_id)}

    def mistakes(self, student_id, limit=20):
        grouped = {}
        for e in self.snapshot(student_id)["events"]:
            if e.get("correct") or not e.get("error"): continue
            k = e["error"].strip().lower()
            x = grouped.setdefault(k, {"pattern": e["error"], "skill": e["skill"], "count": 0, "last_seen": 0})
            x["count"] += 1; x["last_seen"] = max(x["last_seen"], e["timestamp"])
        return sorted(grouped.values(), key=lambda x: (-x["count"], -x["last_seen"]))[:limit]

    def next_problem(self, student_id):
        weakest = self.snapshot(student_id)["weakest"]
        if not weakest: return {"skill": "algebra", "difficulty": "easy", "reason": "diagnostic baseline"}
        s = weakest[0]
        return {"skill": s["skill"], "difficulty": "easy" if s["mastery"] < .4 else "medium" if s["mastery"] < .7 else "hard",
                "reason": f"lowest current mastery ({s['mastery']:.0%})"}

def _slug(value):
    return "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-")[:60]

student_memory = StudentMemoryStore()
