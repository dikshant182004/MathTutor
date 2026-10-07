from __future__ import annotations
import json, os, sys
from urllib.request import Request, urlopen

BASE=os.getenv("V2_API_URL","http://localhost:8000").rstrip("/")
KEY=os.getenv("V2_API_KEY","")

def get(path):
    headers={"X-V2-API-Key":KEY} if KEY else {}
    with urlopen(Request(BASE+path,headers=headers),timeout=10) as r: return json.loads(r.read())

def main():
    print("health:",get("/health"))
    print("plan:",get("/v2/plan?problem=Solve%20x%2B1%3D2"))
    print("snapshot:",get("/v2/students/smoke-student/snapshot"))
    print("graph:",get("/v2/students/smoke-student/graph"))
    print("practice:",get("/v2/students/smoke-student/practice?count=2"))
    print("traces:",get("/v2/observability/traces?limit=3"))
    print("Smoke read-path checks passed.")

if __name__=="__main__": main()
