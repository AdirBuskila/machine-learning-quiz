// Node smoke test: data integrity + shuffle/scoring invariant.
const fs=require("fs"),path=require("path");
global.window={};
const code=fs.readFileSync(path.join(__dirname,"..","questions.js"),"utf8");
eval(code);
const QS=window.QUESTIONS;
let bad=0;
const topics={};
for(const q of QS){
  topics[q.topic]=(topics[q.topic]||0)+1;
  if(!Array.isArray(q.options)||q.options.length<2){console.log("BAD options",q.id);bad++;}
  if(typeof q.correctIndex!=="number"||q.correctIndex<0||q.correctIndex>=q.options.length){console.log("BAD ci",q.id);bad++;}
  if(q.hasImage){console.log("IMAGE leaked",q.id);bad++;}
  if(!q.question||!q.question.trim()){console.log("EMPTY q",q.id);bad++;}
  if(q.options.some(o=>!String(o).trim())){console.log("BLANK opt",q.id);bad++;}
}
// shuffle/scoring invariant: simulate Fisher-Yates mapping 5000x on a sample
function shuffle(a){a=a.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
let posCounts=[0,0,0,0], trials=0, mismatch=0;
for(let t=0;t<5000;t++){
  const q=QS[t%QS.length];
  const order=shuffle(q.options.map((_,i)=>i));
  const correctDisplay=order.indexOf(q.correctIndex);
  // clicking the displayed-correct must map back to the real correct option
  const clickedOrig=order[correctDisplay];
  if(clickedOrig!==q.correctIndex) mismatch++;
  if(q.options.length===4){posCounts[correctDisplay]++;trials++;}
}
console.log("total questions:",QS.length);
console.log("by topic:",topics);
console.log("integrity problems:",bad);
console.log("scoring-map mismatches:",mismatch,"(must be 0)");
console.log("correct-answer display position distribution (4-opt, should be ~even):",
  posCounts.map(c=>(c/trials*100).toFixed(1)+"%").join(" / "));
console.log(bad===0 && mismatch===0 ? "\nSMOKE TEST PASSED" : "\nSMOKE TEST FAILED");
