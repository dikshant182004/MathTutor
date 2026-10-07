from __future__ import annotations
import json, os, sys
from urllib.request import Request, urlopen

BASE=os.getenv("V2_API_URL","http://localhost:8000").rstrip("/")
KEY=os.getenv("V2_API_KEY","")

def request(path, method="GET", body=None):
    headers={"X-V2-API-Key":KEY} if KEY else {}
    if body is not None: headers["Content-Type"]="application/json"
    data=json.dumps(body).encode() if body is not None else None
    with urlopen(Request(BASE+path,headers=headers,method=method,data=data),timeout=10) as r:
        return json.loads(r.read())

def main():
    print("health:",request("/health"))
    print("plan:",request("/v2/plan","POST",{"student_id":"smoke-student","problem":"Solve x+1=2"}))
    print("snapshot:",request("/v2/students/smoke-student/snapshot"))
    print("graph:",get("/v2/students/smoke-student/graph"))
    print("practice:",get("/v2/students/smoke-student/practice?count=2"))
    print("traces:",request("/v2/observability/traces?limit=3"))
    print("Smoke read-path checks passed.")

if __name__=="__main__": main()
