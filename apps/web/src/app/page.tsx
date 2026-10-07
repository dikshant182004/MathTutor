"use client";

import { FormEvent, useEffect, useState } from "react";

type Skill={skill:string;mastery:number;attempts:number;correct:number;status:string};
type GraphNode={id:string;label:string;type:string;mastery?:number;count?:number;status?:string};
type GraphData={nodes:Record<string,GraphNode>;edges:{source:string;target:string;type:string}[]};
type Snapshot={skills:Skill[];weakest:Skill[];events:any[];graph:GraphData};

const API=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000";
const pct=(n:number)=>Math.round(n*100);

function KnowledgeGraph({graph}:{graph:GraphData}){
 const nodes=Object.values(graph?.nodes||{});
 if(!nodes.length)return <div className="empty">Solve a problem to grow your knowledge graph.</div>;
 const pos=nodes.map((n,i)=>({...n,x:90+(i%4)*165,y:70+Math.floor(i/4)*105}));
 const byId=new Map(pos.map(n=>[n.id,n]));
 return <svg className="knowledge-graph" viewBox="0 0 680 320" aria-label="Student knowledge graph">
  {graph.edges.map((e,i)=>{const a=byId.get(e.source),b=byId.get(e.target);return a&&b?<line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="edge"/>:null})}
  {pos.map(n=><g key={n.id} transform={`translate(${n.x},${n.y})`}>
   <circle r={n.type==="mistake"?24:30} className={`node node-${n.status||n.type}`}/>
   <text textAnchor="middle" y="4" className="node-label">{n.label.slice(0,14)}</text>
   {typeof n.mastery==="number"&&<text textAnchor="middle" y="47" className="node-meta">{pct(n.mastery)}%</text>}
  </g>)}
 </svg>;
}

export default function Home(){
 const [studentId]=useState("local-student");
 const [tab,setTab]=useState("Workspace");
 const [problem,setProblem]=useState("");
 const [mode,setMode]=useState<"socratic"|"solve">("socratic");
 const [snapshot,setSnapshot]=useState<Snapshot|null>(null);
 const [mistakes,setMistakes]=useState<any[]>([]);
 const [next,setNext]=useState<any>(null);
 const [answer,setAnswer]=useState("");
 const [loading,setLoading]=useState(false);
 const [trace,setTrace]=useState<string[]>([]);
 const [hintLevel,setHintLevel]=useState(0);

 async function refresh(){
  const [s,m,n]=await Promise.all([
   fetch(`${API}/v2/students/${studentId}/snapshot`).then(r=>r.json()),
   fetch(`${API}/v2/students/${studentId}/mistakes`).then(r=>r.json()),
   fetch(`${API}/v2/students/${studentId}/next-problem`).then(r=>r.json())
  ]);
  setSnapshot(s);setMistakes(m.items||[]);setNext(n);
 }
 useEffect(()=>{refresh().catch(()=>undefined)},[]);

 async function submit(e:FormEvent){
  e.preventDefault();if(!problem.trim()||loading)return;
  setLoading(true);setAnswer("");setTrace(["Input accepted","Adaptive plan selected"]);
  try{
   if(mode==="socratic"){
    const r=await fetch(`${API}/v2/socratic`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({student_id:studentId,problem,skill:next?.skill||"general",hint_level:hintLevel})});
    const d=await r.json();setAnswer(d.prompt||"What relationship can you identify first?");
    setTrace(t=>[...t,"Socratic policy generated","Waiting for student reasoning"]);
   }else{
    setTrace(t=>[...t,"Running solver","Deterministic verification + LLM verification"]);
    const r=await fetch(`${API}/v2/solve`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({student_id:studentId,problem})});
    const d=await r.json();setAnswer(d.final_response||"The session needs a follow-up response.");
    setTrace(t=>[...t,"Verification completed","Student model updated"]);await refresh();
   }
  }catch{setAnswer("The V2 API is unavailable. Start FastAPI and try again.");setTrace(t=>[...t,"Request failed safely"])}
  finally{setLoading(false)}
 }

 const skills=snapshot?.skills||[];const graph=snapshot?.graph||{nodes:{},edges:[]};
 return <main className="shell">
  <aside className="sidebar">
   <div className="brand"><span className="logo">∑</span><div><strong>MathTutor</strong><small>Adaptive Learning V2</small></div></div>
   <nav>{["Workspace","Practice","Mastery","Mistake Lab","Knowledge Graph"].map(x=><button key={x} className={tab===x?"active":""} onClick={()=>setTab(x)}>{x}</button>)}</nav>
   <div className="profile"><div className="avatar">D</div><div><strong>Student</strong><small>{studentId}</small></div></div>
  </aside>
  <section className="workspace">
   <header><div><span className="eyebrow">ADAPTIVE MATHEMATICS AGENT</span><h1>{tab==="Workspace"?"What are you working on?":tab}</h1><p>Diagnosis, guided reasoning, verified solutions and targeted practice.</p></div><div className="status"><i/> V2 agent ready</div></header>

   {tab==="Workspace"&&<><div className="grid">
    <form className="card composer" onSubmit={submit}>
     <div className="mode-row"><button type="button" className={mode==="socratic"?"pill selected":"pill"} onClick={()=>setMode("socratic")}>Socratic tutor</button><button type="button" className={mode==="solve"?"pill selected":"pill"} onClick={()=>setMode("solve")}>Solve & explain</button></div>
     <textarea value={problem} onChange={e=>setProblem(e.target.value)} placeholder="Ask a math question or paste a problem..."/>
     {mode==="socratic"&&<div className="hint-row"><span>Hint level {hintLevel}/3</span><button type="button" onClick={()=>setHintLevel(Math.min(3,hintLevel+1))}>Increase hint</button></div>}
     <div className="composer-footer"><span>Adaptive route · verified math</span><button className="send" disabled={!problem.trim()||loading}>{loading?"Thinking…":"Start learning →"}</button></div>
     {answer&&<div className="response"><span className="tag">{mode==="socratic"?"SOCRATIC STEP":"VERIFIED RESPONSE"}</span><p>{answer}</p></div>}
    </form>
    <section className="card"><div className="card-title"><h2>Mastery</h2><button className="link" onClick={()=>setTab("Mastery")}>View all →</button></div>
     {skills.slice(0,4).map(s=><div className="skill" key={s.skill}><div><strong>{s.skill}</strong><span>{pct(s.mastery)}%</span></div><div className="bar"><i style={{width:`${pct(s.mastery)}%`}}/></div></div>)}
     {!skills.length&&<div className="empty">No skill history yet.</div>}
     <div className="next"><small>NEXT BEST ACTION</small><strong>{next?.skill||"Diagnostic practice"}</strong><p>{next?.reason||"Start a problem to establish your baseline."}</p></div>
    </section>
   </div>
   <div className="bottom-grid">
    <section className="card"><div className="card-title"><h2>Learning loop</h2><span>Live</span></div><div className="steps"><b>1<br/><small>Diagnose</small></b><b>→</b><b>2<br/><small>Teach</small></b><b>→</b><b>3<br/><small>Practice</small></b><b>→</b><b>4<br/><small>Master</small></b></div></section>
    <section className="card"><div className="card-title"><h2>Agent trace</h2><span>Real events</span></div>{(trace.length?trace:["Waiting for a learning task"]).map((x,i)=><div className="activity" key={i}><span>{i<trace.length-1?"✓":"•"}</span>{x}<em>event</em></div>)}</section>
   </div></>}

   {tab==="Mastery"&&<section className="card wide"><div className="card-title"><h2>Student mastery model</h2><button className="link" onClick={refresh}>Refresh</button></div>{skills.map(s=><div className="mastery-row" key={s.skill}><div><strong>{s.skill}</strong><span>{s.attempts} attempts · {s.correct} correct</span></div><div className="bar"><i style={{width:`${pct(s.mastery)}%`}}/></div><b>{pct(s.mastery)}%</b></div>)}{!skills.length&&<div className="empty">Complete a verified problem to populate mastery.</div>}</section>}

   {tab==="Mistake Lab"&&<section className="card wide"><div className="card-title"><h2>Recurring misconceptions</h2><span>Evidence from attempts</span></div>{mistakes.map((m,i)=><div className="mistake" key={i}><div className="mistake-icon">!</div><div><strong>{m.pattern}</strong><span>{m.skill} · seen {m.count} time{m.count===1?"":"s"}</span></div><button onClick={()=>{setProblem(`Practice ${m.skill} focusing on: ${m.pattern}`);setTab("Workspace")}}>Practice →</button></div>)}{!mistakes.length&&<div className="empty">No recurring mistakes detected yet.</div>}</section>}

   {tab==="Practice"&&<section className="card wide practice"><div className="card-title"><h2>Next-best practice</h2><span>Mastery-driven</span></div><div className="practice-card"><small>RECOMMENDED SKILL</small><h2>{next?.skill||"Algebra"}</h2><p>{next?.reason||"Build your first mastery signal."}</p><button className="send" onClick={()=>{setProblem(`Give me a ${next?.difficulty||"easy"} ${next?.skill||"algebra"} problem`);setTab("Workspace")}}>Generate practice →</button></div></section>}

   {tab==="Knowledge Graph"&&<section className="card wide"><div className="card-title"><h2>Knowledge graph</h2><span>{Object.keys(graph.nodes||{}).length} nodes</span></div><KnowledgeGraph graph={graph}/><div className="legend"><span>mastered</span><span>developing</span><span>weak</span><span>misconception</span></div></section>}
  </section>
 </main>;
}