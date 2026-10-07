from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from backend.v2.routing import classify_problem
from backend.v2.socratic import next_socratic_step
from backend.v2.student_memory import StudentMemoryStore
from backend.v2.document_store import PersistentDocumentStore

def load_cases(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def exact_answer_accuracy(cases: list[dict], predict: Callable[[dict], str]) -> float:
    if not cases: return 0.0
    return sum(predict(c).strip()==c["answer"].strip() for c in cases)/len(cases)

def router_regression(cases: list[dict]) -> dict:
    checks=[]
    for c in cases:
        plan=classify_problem(c["problem"],has_student_notes=c.get("has_student_notes",False),explicit_web=c.get("explicit_web",False))
        checks.append(plan.tier.value==c["tier"] and plan.use_rag==c.get("use_rag",plan.use_rag))
    return {"accuracy":sum(checks)/len(checks) if checks else 0.0,"cases":len(checks)}

def socratic_regression(cases: list[dict]) -> dict:
    checks=[]
    for c in cases:
        step=next_socratic_step(problem=c["problem"],student_attempt=c.get("attempt",""),skill=c.get("skill","general"),hint_level=c.get("hint_level",0))
        checks.append(step.kind==c["expected_kind"])
    return {"accuracy":sum(checks)/len(checks) if checks else 0.0,"cases":len(checks)}

def memory_regression(root: str) -> dict:
    store=StudentMemoryStore(str(Path(root)/"memory.json"))
    store.record_attempt("student-a","algebra",correct=True)
    store.record_attempt("student-b","algebra",correct=False,error="sign error")
    return {
        "isolation": store.mistakes("student-a")==[] and store.mistakes("student-b")[0]["count"]==1,
        "mastery_moves": store.get_skill("student-a","algebra").mastery > .5,
    }

def rag_regression(root: str) -> dict:
    store=PersistentDocumentStore(str(Path(root)/"docs.json"))
    store.ingest("student-a","notes","integration by parts and substitution","notes.pdf",3)
    hit=store.retrieve("student-a","integration by parts")
    leak=store.retrieve("student-b","integration by parts")
    return {"hit":bool(hit),"provenance":bool(hit and hit[0]["metadata"]["page"]==3),"isolation":not leak}

def run_all() -> dict:
    return {
        "router": router_regression(load_cases("evals/datasets/router.jsonl")),
        "socratic": socratic_regression(load_cases("evals/datasets/socratic.jsonl")),
        "memory": memory_regression(".data/eval"),
        "rag": rag_regression(".data/eval"),
    }

if __name__ == "__main__":
    import pprint
    pprint.pp(run_all())
