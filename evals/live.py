from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from backend.agents.graph import chatbot
from backend.agents.state import make_initial_state

def extract_final(text: str) -> str:
    matches=re.findall(r"(?:Final Answer|answer is)\s*[:=]\s*(?:\\boxed\{)?([^}\n]+)", text, re.I)
    return matches[-1].strip().strip("$ .") if matches else text.strip().splitlines()[-1].strip()

def evaluate(path: str, limit: int, student_id: str) -> dict:
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()][:limit]
    results=[]; started=time.perf_counter()
    for i,row in enumerate(rows):
        problem=row.get("question") or row.get("problem") or row.get("prompt")
        expected=row.get("answer") or row.get("target") or ""
        state=make_initial_state(student_id=student_id,thread_id=f"eval-{i}",raw_text=problem)
        t=time.perf_counter()
        try:
            out=chatbot.invoke(state,config={"configurable":{"thread_id":f"eval-{i}"}})
            actual=extract_final(out.get("final_response") or "")
            ok=actual==expected.strip()
            results.append({"index":i,"correct":ok,"latency_ms":round((time.perf_counter()-t)*1000,2)})
        except Exception as exc:
            results.append({"index":i,"correct":False,"error":type(exc).__name__,"latency_ms":round((time.perf_counter()-t)*1000,2)})
    return {
        "dataset":path,"cases":len(results),
        "accuracy":sum(r["correct"] for r in results)/len(results) if results else 0.0,
        "elapsed_ms":round((time.perf_counter()-started)*1000,2),
        "cases_detail":results,
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run sampled MathTutor V2 live workflow evaluation.")
    parser.add_argument("dataset"); parser.add_argument("--limit",type=int,default=25); parser.add_argument("--student-id",default="eval-student")
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    report=evaluate(args.dataset,args.limit,args.student_id)
    text=json.dumps(report,indent=2)
    print(text)
    if args.output: Path(args.output).write_text(text,encoding="utf-8")
