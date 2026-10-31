"""Module 7 - Build a self-contained interactive dashboard.html (no server needed; works offline)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, plotly
from config import *

TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAC Care-Load Forecast Desk</title>
<style>
:root{--navy:#0B1F3A;--teal:#0F9D8A;--amber:#F59E0B;--red:#E11D48;--indigo:#4F46E5;--slate:#64748B;--bg:#F4F6FA;--card:#fff;--line:#E2E8F0}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:#1E293B}
.wrap{display:grid;grid-template-columns:270px 1fr;min-height:100vh}
aside{background:var(--navy);color:#E2E8F0;padding:22px 18px;position:sticky;top:0;height:100vh;overflow:auto}
aside h3{margin:0 0 14px;font-size:1.05rem;color:#fff}aside label{display:block;font-size:.78rem;margin:16px 0 6px;color:#94A3B8;text-transform:uppercase;letter-spacing:.05em}
aside select,aside input[type=range]{width:100%}aside select{padding:7px;border-radius:8px;border:0}
.seg{display:flex;gap:6px}.seg button{flex:1;padding:7px 0;border-radius:8px;border:1px solid #334155;background:#14294A;color:#E2E8F0;cursor:pointer;font-size:.85rem}
.seg button.on{background:var(--teal);border-color:var(--teal);color:#fff;font-weight:600}
.chk{display:flex;align-items:center;gap:8px;font-size:.84rem;margin:4px 0;text-transform:none;letter-spacing:0;color:#E2E8F0}
.small{font-size:.76rem;color:#94A3B8;margin-top:10px;line-height:1.4}
main{padding:20px 26px;min-width:0}
.hero{border-radius:16px;padding:20px 26px;color:#fff;margin-bottom:14px}.hero h2{margin:0 0 4px;font-size:1.5rem}.hero p{margin:0;font-size:1rem;opacity:.95}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--teal);border-radius:12px;padding:12px 14px}
.kpi .l{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--slate);font-weight:600}.kpi .v{font-size:1.8rem;font-weight:700;color:var(--navy)}.kpi .s{font-size:.76rem;color:var(--slate)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.tabs button{padding:9px 16px;border-radius:999px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:.9rem}
.tabs button.on{background:var(--navy);color:#fff;border-color:var(--navy)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}.row3{display:grid;grid-template-columns:3fr 2fr;gap:14px}
table{border-collapse:collapse;width:100%;font-size:.84rem}th{background:var(--navy);color:#fff;padding:7px 9px;text-align:center}td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:center}
td:first-child,th:first-child{text-align:left}tr.hl td{background:#E6F6F3;font-weight:600}
.note{background:#F1F5F9;border-radius:10px;padding:10px 14px;font-size:.88rem;color:#334155;margin:10px 0}
.ctl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:.88rem}.ctl select,.ctl input{padding:5px}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}aside{position:static;height:auto}.kpis{grid-template-columns:repeat(2,1fr)}.row,.row3{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap">
<aside>
<h3>Control rail</h3>
<label>Forecast horizon</label><div class="seg" id="hseg"></div>
<label>Primary model</label><select id="model"></select>
<label>Overlay other models</label><div id="overlay"></div>
<label>Uncertainty band</label><div class="seg" id="bseg"></div>
<label>Capacity headroom over today's load: <span id="hv">5</span>%</label><input type="range" id="head" min="0" max="30" value="5">
<div class="small" id="capinfo"></div>
<div class="small" id="orig"></div>
</aside>
<main>
<div class="hero" id="hero"></div>
<div class="kpis" id="kpis"></div>
<div class="tabs" id="tabs"></div>
<div id="view"></div>
</main></div>
<script>__PLOTLY__</script>
<script>
const D = __DATA__;
const MODELS=["Naive","Moving Avg","ARIMA","SARIMA","Exp. Smoothing","Random Forest","Gradient Boosting"];
const FAM={"Naive":"Baseline","Moving Avg":"Baseline","ARIMA":"Statistical","SARIMA":"Statistical","Exp. Smoothing":"Statistical","Random Forest":"Machine learning","Gradient Boosting":"Machine learning"};
const COL={"Naive":"#94A3B8","Moving Avg":"#A78BFA","ARIMA":"#0EA5E9","SARIMA":"#0369A1","Exp. Smoothing":"#14B8A6","Random Forest":"#F97316","Gradient Boosting":"#E11D48"};
const TL={hhs:"Children in HHS care",discharges:"Discharges to sponsors",transfers:"Transfers into HHS"};
const NAVY="#0B1F3A",TEAL="#0F9D8A",AMBER="#F59E0B",RED="#E11D48",INDIGO="#4F46E5";
const S={H:14,model:D.best,overlay:[],band:"80",head:5,tab:"care",lk1:120,lk2:90,tg:"hhs",mt:"MAE",hz:7,tg4:"hhs",up:150,dn:-25};
const $=id=>document.getElementById(id);
const fmt=(x,d=0)=>Number(x).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const cap=()=>D.cur*(1+S.head/100);
const mean=a=>a.reduce((s,v)=>s+v,0)/Math.max(a.length,1);
const sum=a=>a.reduce((s,v)=>s+v,0);
const P=(t,m,H)=>{const o=D.fc[t][m];const k=n=>o[n].slice(0,H);return{x:k("dates"),f:k("forecast"),lo80:k("lo80"),hi80:k("hi80"),lo95:k("lo95"),hi95:k("hi95"),scale:o.scale}};
const HS=[1,3,7,14,21,28];
function breach(model,capv,H){const p=P("hhs",model,H);return p.f.map((f,i)=>{const h=i+1;const hh=HS.reduce((b,x)=>Math.abs(x-h)<Math.abs(b-h)?x:b,HS[0]);const r=D.resid[model][hh];return r.filter(e=>f+e*p.scale>capv).length/r.length;});}
function histWin(days){const n=D.hist.dates.length;const s=Math.max(0,n-days);return{x:D.hist.dates.slice(s),hhs:D.hist.hhs.slice(s),dis:D.hist.dis.slice(s),tra:D.hist.tra.slice(s)};}
function roll(a,w,minp){return a.map((_,i)=>{const s=a.slice(Math.max(0,i-w+1),i+1);return s.length>=(minp||w)?mean(s):null});}
const lay=(o={})=>Object.assign({margin:{l:55,r:15,t:26,b:40},height:420,hovermode:"x unified",legend:{orientation:"h",y:-0.2},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Segoe UI,Arial",size:12}},o);
const plot=(id,tr,l)=>Plotly.react(id,tr,l,{responsive:true,displaylogo:false,displayModeBar:false});
const bandTr=(p,name,rgba)=>{const lo=S.band==="80"?p.lo80:p.lo95,hi=S.band==="80"?p.hi80:p.hi95;return{x:p.x.concat(p.x.slice().reverse()),y:hi.concat(lo.slice().reverse()),fill:"toself",fillcolor:rgba,line:{width:0},hoverinfo:"skip",name:name+" "+S.band+"% band"};};
const capShape=(c,x0,x1)=>({type:"line",x0:x0,x1:x1,y0:c,y1:c,line:{color:RED,dash:"dash",width:1.5}});
const originShape=()=>({type:"line",x0:D.origin,x1:D.origin,y0:0,y1:1,yref:"paper",line:{color:"#64748B",dash:"dot",width:1}});
function tbl(head,rows,hl){return"<table><tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>"+rows.map(r=>"<tr"+(hl&&hl(r)?' class="hl"':"")+">"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>").join("")+"</table>";}

/* ---------- controls ---------- */
function seg(id,opts,get,set){const el=$(id);el.innerHTML="";opts.forEach(o=>{const b=document.createElement("button");b.textContent=o.label;b.className=get()===o.v?"on":"";b.onclick=()=>{set(o.v);seg(id,opts,get,set);update();};el.appendChild(b);});}
function initControls(){
 seg("hseg",[7,14,21,28].map(v=>({v,label:v+"d"})),()=>S.H,v=>S.H=v);
 seg("bseg",[{v:"80",label:"80%"},{v:"95",label:"95%"}],()=>S.band,v=>S.band=v);
 $("model").innerHTML=MODELS.map(m=>`<option ${m===S.model?"selected":""}>${m}</option>`).join("");
 $("model").onchange=e=>{S.model=e.target.value;S.overlay=S.overlay.filter(m=>m!==S.model);drawOverlay();update();};
 $("head").oninput=e=>{S.head=+e.target.value;update();};
 drawOverlay();
 $("tabs").innerHTML=[["care","Care load forecast"],["dis","Discharge demand"],["lab","Model lab"],["ci","Confidence intervals"],["scen","Scenario comparison"]].map(t=>`<button data-t="${t[0]}">${t[1]}</button>`).join("");
 $("tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.tab=b.dataset.t;update();});
}
function drawOverlay(){$("overlay").innerHTML=MODELS.filter(m=>m!==S.model).map(m=>`<label class="chk"><input type="checkbox" ${S.overlay.includes(m)?"checked":""} data-m="${m}"> ${m}</label>`).join("");
 $("overlay").querySelectorAll("input").forEach(i=>i.onchange=()=>{const m=i.dataset.m;S.overlay=i.checked?S.overlay.concat([m]):S.overlay.filter(x=>x!==m);update();});}

/* ---------- header ---------- */
function header(){
 const ph=P("hhs",S.model,S.H),pd=P("discharges",S.model,S.H),pt=P("transfers",S.model,S.H);
 const pb=breach(S.model,cap(),S.H),pmax=Math.max(...pb);
 const [status,color]=pmax<0.2?["STABLE",TEAL]:pmax<0.5?["WATCH",AMBER]:["ALERT",RED];
 const i=S.H-1,lo=S.band==="80"?ph.lo80[i]:ph.lo95[i],hi=S.band==="80"?ph.hi80[i]:ph.hi95[i];
 const net=sum(pt.f)-sum(pd.f);
 $("hero").style.background=`linear-gradient(120deg,${NAVY},${color}CC)`;
 $("hero").innerHTML=`<h2>Capacity outlook: ${status}</h2><p>In ${S.H} days the care load is projected at <b>${fmt(ph.f[i])}</b> children (${S.band}% range ${fmt(lo)} to ${fmt(hi)}) versus ${fmt(D.cur)} today. Chance of exceeding ${fmt(cap())} at some point: <b>${fmt(pmax*100)}%</b>. Projected transfers in minus discharges out: <b>${net>=0?"+":""}${fmt(net)}</b> children over the window.</p>`;
 $("capinfo").innerHTML=`Capacity threshold = <b>${fmt(cap())}</b> children (today's load ${fmt(D.cur)}).`;
 $("orig").innerHTML=`Forecast origin: ${D.origin} (latest reported day).`;$("hv").textContent=S.head;
 const sg=D.sg[S.model],m7=D.mh.hhs[S.model].MAPE["7"];
 const k=(l,v,s,c)=>`<div class="kpi" style="border-top-color:${c||TEAL}"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
 $("kpis").innerHTML=k("Forecast accuracy",fmt(100-m7,1)+"%","care load, 7-day horizon (100 - MAPE)")+
  k("Surge lead time",sg.lead==null?"n/a":fmt(sg.lead)+" days",`avg warning; caught ${sg.hits} of ${sg.events} stress events`,INDIGO)+
  k("Capacity breach probability",fmt(pmax*100)+"%",`within ${S.H} days at ${fmt(cap())}`,color)+
  k("Forecast stability index",fmt(D.stab[S.model],1),"100 = forecasts barely change week to week")+
  k("Model robustness",fmt(D.rob[S.model])+"%","of backtest folds beating naive persistence",INDIGO);
}

/* ---------- tabs ---------- */
function viewCare(){
 $("view").innerHTML=`<div class="card"><div class="ctl">History shown: <input type="range" id="lk" min="30" max="365" value="${S.lk1}"> <span id="lkv">${S.lk1}</span> days</div><div id="c1"></div></div><div class="card" id="t1"></div>`;
 $("lk").oninput=e=>{S.lk1=+e.target.value;viewCare();};
 const h=histWin(S.lk1),p=P("hhs",S.model,S.H);
 const tr=[{x:h.x,y:h.hhs,name:"Reported care load",line:{color:NAVY,width:2.4}},bandTr(p,S.model,"rgba(15,157,138,.2)"),{x:p.x,y:p.f,name:S.model+" forecast",line:{color:COL[S.model],width:3}}];
 S.overlay.forEach(m=>{const o=P("hhs",m,S.H);tr.push({x:o.x,y:o.f,name:m,line:{color:COL[m],width:1.6,dash:"dash"}});});
 plot("c1",tr,lay({height:470,yaxis:{title:"Children in HHS care"},shapes:[capShape(cap(),h.x[0],p.x[p.x.length-1]),originShape()],annotations:[{x:h.x[0],y:cap(),text:"capacity "+fmt(cap()),showarrow:false,yshift:10,xanchor:"left",font:{color:RED,size:11}}]}));
 const full=P("hhs",S.model,28),B=S.band;
 $("t1").innerHTML=tbl(["Days ahead","Date","Forecast","Low ("+B+"%)","High ("+B+"%)","Change vs today"],[7,14,21,28].map(d=>{const i=d-1;return[d,full.x[i],fmt(full.f[i]),fmt(full["lo"+B][i]),fmt(full["hi"+B][i]),(full.f[i]-D.cur>=0?"+":"")+fmt(full.f[i]-D.cur)];}));
}
function viewDis(){
 $("view").innerHTML=`<div class="card"><div class="ctl">History shown: <input type="range" id="lk" min="30" max="365" value="${S.lk2}"> <span>${S.lk2}</span> days</div><div id="c2"></div></div><div class="row3"><div class="card" id="t2"></div><div class="card" id="v2"></div></div>`;
 $("lk").onchange=e=>{S.lk2=+e.target.value;viewDis();};
 const h=histWin(S.lk2),pd=P("discharges",S.model,S.H),pt=P("transfers",S.model,S.H);
 const tr=[{x:h.x,y:roll(h.dis,7),name:"Discharges (7-day avg)",line:{color:TEAL,width:2.4}},{x:h.x,y:roll(h.tra,7),name:"Transfers in (7-day avg)",line:{color:AMBER,width:2.4}},
  bandTr(pd,"Discharge (single-day)","rgba(15,157,138,.18)"),
  {x:pd.x,y:roll(pd.f,7,1),name:"Discharge forecast (7-day avg)",line:{color:TEAL,width:3,dash:"dash"}},
  {x:pt.x,y:roll(pt.f,7,1),name:"Transfers forecast (7-day avg)",line:{color:AMBER,width:3,dash:"dash"}}];
 plot("c2",tr,lay({height:430,yaxis:{title:"Children per day"},shapes:[originShape()]}));
 const rows=[];for(let w=1;w<=S.H/7;w++){const a=(w-1)*7,b=w*7,ti=sum(pt.f.slice(a,b)),dout=sum(pd.f.slice(a,b));rows.push(["Week "+w,fmt(ti),fmt(dout),fmt(ti-dout),fmt(dout/Math.max(ti,1)*100)+"%"]);}
 $("t2").innerHTML="<b>Week-by-week outlook</b>"+tbl(["Week","Transfers in","Discharges out","Net (in - out)","Discharges / transfers"],rows);
 const ts=sum(pt.f),ds=sum(pd.f),extra=D.other/D.beta*S.H,ratio=ds/Math.max(ts+extra,1);
 $("v2").innerHTML=`<div class="note">Over ${S.H} days discharges are projected at <b>${fmt(ds)}</b> versus <b>${fmt(ts)}</b> transfers in. Past data also show about <b>${fmt(D.other/D.beta,1)}</b> children/day entering care outside the transfers column (+${fmt(extra)} over the window), so coverage of total inflow is <b>${fmt(ratio*100)}%</b>. ${ratio>=1?"Discharge capacity is projected to <b>offset</b> total inflow.":"Discharges are projected to <b>fall short</b> of total inflow, so the care load keeps building."}</div>`;
}
function viewLab(){
 const opt=(o,sel)=>Object.entries(o).map(([k,v])=>`<option value="${k}" ${k===sel?"selected":""}>${v}</option>`).join("");
 $("view").innerHTML=`<div class="card"><div class="ctl">Target <select id="tg">${opt(TL,S.tg)}</select> Error metric <select id="mt">${opt({MAE:"MAE",RMSE:"RMSE",MAPE:"MAPE",WAPE:"WAPE"},S.mt)}</select> <span class="small" style="color:#64748B">WAPE = total absolute error / total actual; steadier than MAPE for small daily flows.</span></div><div id="c3"></div></div><div class="card" id="t3"></div><div class="card"><div class="ctl">Backtest horizon to inspect <select id="hz">${[1,3,7,14,21,28].map(h=>`<option ${h===S.hz?"selected":""}>${h}</option>`).join("")}</select> days</div><div id="c3b"></div></div>`;
 $("tg").onchange=e=>{S.tg=e.target.value;viewLab();};$("mt").onchange=e=>{S.mt=e.target.value;viewLab();};$("hz").onchange=e=>{S.hz=+e.target.value;viewLab();};
 plot("c3",MODELS.map(m=>({x:HS,y:HS.map(h=>D.mh[S.tg][m][S.mt][h]),name:m,mode:"lines+markers",line:{color:COL[m],width:m===S.model?4:1.8}})),lay({height:380,hovermode:"closest",xaxis:{title:"Forecast horizon (days)"},yaxis:{title:S.mt}}));
 const rows=MODELS.map(m=>{const o=D.mo[S.tg][m];return[m,FAM[m],fmt(o.MAE,1),fmt(o.RMSE,1),fmt(o.MAPE,1),fmt(o.WAPE,1),fmt(o.rank,2)];}).sort((a,b)=>a[6]-b[6]);
 const fam={};MODELS.forEach(m=>{const f=FAM[m];fam[f]=fam[f]||{s:[],l:[]};HS.forEach(h=>{(h<=7?fam[f].s:fam[f].l).push(D.mh[S.tg][m].MAE[h]);});});
 $("t3").innerHTML=tbl(["Model","Family","MAE","RMSE","MAPE %","WAPE %","Avg rank (1 = best)"],rows,r=>r[0]===S.model)+`<div class="note"><b>Statistical vs machine learning (mean MAE):</b> ${Object.entries(fam).map(([k,v])=>`${k}: ${fmt(mean(v.s),1)} (1-7d), ${fmt(mean(v.l),1)} (14-28d)`).join(" | ")}</div>`;
 const b=D.bt[S.tg][S.model][S.hz];
 plot("c3b",[{x:b.x,y:b.a,name:"Actual",line:{color:NAVY,width:2.4}},{x:b.x,y:b.f,name:S.model+" ("+S.hz+"-day-ahead)",line:{color:COL[S.model],width:2.4,dash:"dash"}}],lay({height:340,yaxis:{title:TL[S.tg]}}));
}
function viewCI(){
 $("view").innerHTML=`<div class="card"><div class="ctl">Target <select id="tg4">${Object.entries(TL).map(([k,v])=>`<option value="${k}" ${k===S.tg4?"selected":""}>${v}</option>`).join("")}</select></div><div id="c4"></div></div><div class="row"><div class="card" id="t4"></div><div class="card"><div id="c4b"></div></div></div>`;
 $("tg4").onchange=e=>{S.tg4=e.target.value;viewCI();};
 const t=S.tg4,p=P(t,S.model,28),h=histWin(60),hv=t==="hhs"?h.hhs:t==="discharges"?h.dis:h.tra;
 const band=(lo,hi,c,n)=>({x:p.x.concat(p.x.slice().reverse()),y:p[hi].concat(p[lo].slice().reverse()),fill:"toself",fillcolor:c,line:{width:0},name:n,hoverinfo:"skip"});
 const tr=[band("lo95","hi95","rgba(79,70,229,.12)","95% interval"),band("lo80","hi80","rgba(79,70,229,.26)","80% interval"),{x:h.x,y:hv,name:"Reported",line:{color:NAVY,width:2}},{x:p.x,y:p.f,name:"Forecast",line:{color:INDIGO,width:3}}];
 const sh=[originShape()];if(t==="hhs")sh.push(capShape(cap(),h.x[0],p.x[27]));
 plot("c4",tr,lay({height:430,yaxis:{title:TL[t]},shapes:sh}));
 const c=D.cov[t][S.model];
 $("t4").innerHTML="<b>Does the band hold? Out-of-sample coverage by horizon</b> (nominal 80% / 95%)"+tbl(["Days ahead","Achieved 80%","Achieved 95%","Forecasts checked"],HS.map(x=>[x,fmt(c[x][0]*100)+"%",fmt(c[x][1]*100)+"%",c[x][3]]));
 if(t==="hhs"){const pb=breach(S.model,cap(),28);plot("c4b",[{x:p.x,y:pb.map(v=>v*100),fill:"tozeroy",line:{color:RED,width:2.5},name:"P(load > capacity)"}],lay({height:300,title:{text:"Breach probability by day (capacity "+fmt(cap())+")",font:{size:13}},yaxis:{title:"%"},showlegend:false}));}
 else plot("c4b",[{x:HS,y:HS.map(x=>c[x][2]),type:"bar",marker:{color:INDIGO}}],lay({height:300,title:{text:"Average 80% interval width (children/day)",font:{size:13}},xaxis:{title:"Days ahead"},showlegend:false,hovermode:"closest"}));
}
function viewScen(){
 $("view").innerHTML=`<div class="note">What-if engine: change in care load per child of net flow = <b>${D.beta.toFixed(2)}</b> (fitted on the last 365 days, weekly changes). Scenario deltas = this factor x cumulative change in (transfers in - discharges out) applied to the primary model's forecast. It is a planning aid, not a causal model.</div>
 <div class="card"><div class="ctl">Custom: change in transfers into HHS <input type="range" id="up" min="-50" max="300" step="10" value="${S.up}"> <b id="upv">${S.up}%</b> &nbsp; change in discharges <input type="range" id="dn" min="-75" max="100" step="5" value="${S.dn}"> <b id="dnv">${S.dn}%</b></div><div id="c5"></div></div><div class="card" id="t5"></div>`;
 $("up").oninput=e=>{S.up=+e.target.value;viewScen();};$("dn").oninput=e=>{S.dn=+e.target.value;viewScen();};
 const p=P("hhs",S.model,S.H),T=P("transfers",S.model,S.H).f,Dd=P("discharges",S.model,S.H).f;
 const sc=[["Baseline",0,0,NAVY],["Intake surge (+100% transfers)",100,0,AMBER],["Faster discharges (+50%)",0,50,TEAL],["Surge + slower discharges (+100% / -25%)",100,-25,RED],["Custom",S.up,S.dn,INDIGO]];
 const tr=[],rows=[];
 sc.forEach(([nm,pu,pdn,c])=>{let cum=0;const y=p.f.map((f,i)=>{cum+=D.beta*((pu/100)*T[i]-(pdn/100)*Dd[i]);return f+cum;});
  tr.push({x:p.x,y,name:nm,line:{color:c,width:nm==="Baseline"?3.4:2.2}});
  const k=y.findIndex(v=>v>cap());rows.push([nm,fmt(y[y.length-1]),(y[y.length-1]-p.f[p.f.length-1]>=0?"+":"")+fmt(y[y.length-1]-p.f[p.f.length-1]),k>=0?p.x[k]:"not within window"]);});
 plot("c5",tr,lay({height:430,yaxis:{title:"Children in HHS care"},shapes:[capShape(cap(),p.x[0],p.x[p.x.length-1])]}));
 $("t5").innerHTML=tbl(["Scenario","Load in "+S.H+" days","vs baseline","First day above capacity"],rows);
}
function update(){
 header();
 $("tabs").querySelectorAll("button").forEach(b=>b.className=b.dataset.t===S.tab?"on":"");
 ({care:viewCare,dis:viewDis,lab:viewLab,ci:viewCI,scen:viewScen})[S.tab]();
 seg("hseg",[7,14,21,28].map(v=>({v,label:v+"d"})),()=>S.H,v=>S.H=v);
}
initControls();update();
</script></body></html>'''


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    d = pd.read_csv(OUT / "daily_clean.csv", parse_dates=["date"], index_col="date")
    fc = pd.read_csv(OUT / "forecast_final.csv", parse_dates=["date"])
    bt = pd.read_csv(OUT / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    bt["rel_resid"] = (bt["actual"] - bt["forecast"]) / bt["scale"]
    mh = pd.read_csv(OUT / "metrics_by_horizon.csv"); mo = pd.read_csv(OUT / "metrics_overall.csv")
    rk = pd.read_csv(OUT / "model_ranking.csv"); cov = pd.read_csv(OUT / "interval_coverage.csv")
    sg = pd.read_csv(OUT / "surge_warning.csv"); st = pd.read_csv(OUT / "stability.csv")
    rb = pd.read_csv(OUT / "robustness.csv")
    kp = json.load(open(OUT / "kpi_summary.json"))
    origin = pd.Timestamp(kp["origin"])
    x = d.iloc[-365:]
    dh, ns = x.hhs.diff(7), (x.transfers - x.discharges).rolling(7).sum()
    ok = dh.notna() & ns.notna()
    b, a = np.polyfit(ns[ok], dh[ok], 1)
    data = {"origin": str(origin.date()), "cur": float(d.loc[origin, "hhs"]), "best": kp["best_overall"],
            "beta": float(np.clip(b, 0.5, 1.5)), "other": float(a) / 7,
            "hist": {"dates": [str(i.date()) for i in x.index], "hhs": [r(v, 0) for v in x.hhs],
                     "dis": [r(v, 1) for v in x.discharges], "tra": [r(v, 1) for v in x.transfers]}}
    data["fc"] = {t: {m: {"dates": [str(v.date()) for v in g.date], "forecast": [r(v, 1) for v in g.forecast],
                          "lo80": [r(v, 1) for v in g.lo80], "hi80": [r(v, 1) for v in g.hi80],
                          "lo95": [r(v, 1) for v in g.lo95], "hi95": [r(v, 1) for v in g.hi95], "scale": r(g.scale.iloc[0], 3)}
                      for m, g in fc[fc.target == t].sort_values("horizon").groupby("model")} for t in TARGETS}
    data["mh"] = {t: {m: {k: {str(int(h)): r(v, 3) for h, v in zip(g.horizon, g[k])} for k in ["MAE", "RMSE", "MAPE", "WAPE"]}
                      for m, g in mh[mh.target == t].groupby("model")} for t in TARGETS}
    mo = mo.merge(rk.groupby(["target", "model"])["rank"].mean().reset_index(), on=["target", "model"])
    data["mo"] = {t: {r_.model: {k: r(getattr(r_, k), 3) for k in ["MAE", "RMSE", "MAPE", "WAPE", "rank"]} for r_ in g.itertuples()}
                  for t, g in mo.groupby("target")}
    data["cov"] = {t: {m: {str(int(r_.horizon)): [r(r_.cover80, 3), r(r_.cover95, 3), r(r_.width80, 2), int(r_.n)] for r_ in g.itertuples()}
                       for m, g in cov[cov.target == t].groupby("model")} for t in TARGETS}
    data["sg"] = {r_.model: {"events": int(r_.events), "hits": int(r_.hits), "lead": r(r_.avg_lead_days, 1)} for r_ in sg.itertuples()}
    data["stab"] = {r_.model: r(r_.stability_index, 2) for r_ in st.itertuples()}
    data["rob"] = {r_.model: r(r_.robustness_win_rate, 1) for r_ in rb[rb.target == "hhs"].itertuples()}
    g = bt[(bt.target == "hhs") & bt.observed]
    data["resid"] = {m: {str(int(h)): [r(v, 4) for v in gg.rel_resid] for h, gg in gm.groupby("horizon")} for m, gm in g.groupby("model")}
    o = bt[bt.observed].sort_values("target_date")
    data["bt"] = {t: {m: {str(int(h)): {"x": [str(v.date()) for v in gg.target_date], "a": [r(v, 1) for v in gg.actual], "f": [r(v, 1) for v in gg.forecast]}
                          for h, gg in gm.groupby("horizon")} for m, gm in o[o.target == t].groupby("model")} for t in TARGETS}
    js = open(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", encoding="utf-8").read()
    html = TEMPLATE.replace("__PLOTLY__", js).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "dashboard.html").write_text(html, encoding="utf-8")
    print("dashboard.html written", round(len(html) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
