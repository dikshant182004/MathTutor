from __future__ import annotations

import random
from dataclasses import dataclass

@dataclass(frozen=True)
class PracticeProblem:
    skill: str
    difficulty: str
    prompt: str
    answer: str

def generate_problem(skill: str, difficulty: str, seed: int | None = None) -> PracticeProblem:
    rng=random.Random(seed)
    skill=(skill or "algebra").lower().replace("_"," ")
    difficulty=difficulty if difficulty in {"easy","medium","hard"} else "easy"
    if "linear" in skill or skill=="algebra":
        if difficulty=="easy":
            a=rng.randint(2,9); x=rng.randint(1,12); b=rng.randint(-9,9); c=a*x+b
            return PracticeProblem(skill,difficulty,f"Solve {a}x {b:+d} = {c}.",str(x))
        if difficulty=="medium":
            x=rng.randint(-8,8); a=rng.randint(2,7); b=rng.randint(-9,9); c=a*x+b
            return PracticeProblem(skill,difficulty,f"Solve {a}(x {b:+d}) = {a*(x+b)}.",str(x))
        x=rng.randint(-6,6); return PracticeProblem(skill,difficulty,f"Solve x² {(-2*x):+d}x {x*x:+d} = 0.",f"{x} (double root)")
    if "derivative" in skill or "calculus" in skill:
        n=rng.randint(2,6); p=rng.randint(1,4)
        return PracticeProblem(skill,difficulty,f"Differentiate f(x) = {n}x^{p}.",f"{n*p}x^{p-1}")
    if "probability" in skill:
        total=rng.randint(4,12); favorable=rng.randint(1,total-1)
        return PracticeProblem(skill,difficulty,f"A fair outcome has {favorable} favorable cases out of {total}. What is the probability?",f"{favorable}/{total}")
    return PracticeProblem(skill,difficulty,"Simplify 3x + 2x - x.","4x")

def generate_session(skill: str, difficulty: str, count: int = 5, seed: int | None = None):
    count=max(1,min(count,20))
    return [generate_problem(skill,difficulty,None if seed is None else seed+i).__dict__ for i in range(count)]
