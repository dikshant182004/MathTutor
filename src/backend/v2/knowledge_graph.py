from __future__ import annotations

PREREQUISITES = {
    "calculus": ["algebra", "functions"],
    "derivatives": ["algebra", "functions"],
    "integrals": ["algebra", "derivatives"],
    "probability": ["arithmetic"],
    "linear_equations": ["arithmetic"],
    "quadratics": ["algebra"],
    "geometry": ["arithmetic"],
}

def prerequisite_edges(skill: str):
    return [{"source": "skill:"+p, "target": "skill:"+skill, "type": "prerequisite"} for p in PREREQUISITES.get(skill.lower(), [])]
