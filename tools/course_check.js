// Guided courses ("צריך עזרה?"): data lint + index.html script-tag sync.
// usage: node tools/course_check.js            lint every course, fail on any error
//        node tools/course_check.js --sync     also rewrite the COURSE LEVELS script block in index.html
//        node tools/course_check.js --dump cls [C03]   print the questions with their keyed answer, for review
const fs = require("fs"), path = require("path"), vm = require("vm");
const ROOT = path.join(__dirname, "..");
const args = process.argv.slice(2);

const COURSE_TOPICS = { cls: ["knn", "trees", "naive_bayes"], eval: ["intro", "evaluation"],
  reg: ["regression", "neural_nets", "svm"], unsup: ["clustering", "text", "image", "ensemble", "flow"] };
const PREFIX = { cls: "C", eval: "E", reg: "R", unsup: "U" };

const files = Object.keys(COURSE_TOPICS).flatMap(c => {
  const dir = path.join(ROOT, "courses", c);
  return fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => /\.js$/.test(f)).sort().map(f => `courses/${c}/${f}`) : [];
});

const ctx = { console }; ctx.window = ctx;   // browser globals: window.X and bare X are the same
vm.createContext(ctx);
const load = f => vm.runInContext(fs.readFileSync(path.join(ROOT, f), "utf8"), ctx, { filename: f });
load("questions.js"); load("courses.js");
files.forEach(load);
const BANK = ctx.window.QUESTIONS, byId = new Map(BANK.map(q => [q.id, q]));
const COURSES = ctx.window.COURSES;

let bad = 0; const err = (...a) => { console.log("ERROR", ...a); bad++; };
const warn = (...a) => console.log("warn ", ...a);
const QID = /\b((?:\d{2}[ABS]-[ABC])|SAMP-\d|PRAC-[NX])-Q(\d+)\b/g;
const refsIn = s => [...String(s || "").matchAll(QID)].map(m => m[0]);
const HEB = /[֐-׿]/;

const qIds = new Set(); let total = 0, totalRefs = 0, cited = new Set();
Object.keys(COURSE_TOPICS).forEach(cid => {
  const LV = (COURSES[cid] || { levels: [] }).levels;
  if (!LV.length) { warn(cid, "has no levels yet"); return; }
  const lvIds = new Set(), orders = new Set();
  LV.forEach((L, i) => {
    const W = `${cid}/${L.id}`;
    if (!new RegExp(`^${PREFIX[cid]}\\d\\d$`).test(L.id)) err(W, `level id must look like ${PREFIX[cid]}01`);
    if (lvIds.has(L.id)) err(W, "duplicate level id"); lvIds.add(L.id);
    if (orders.has(L.order)) err(W, "duplicate order"); orders.add(L.order);
    if (L.order !== i + 1) err(W, `order ${L.order} but position ${i + 1} (orders must be 1..n)`);
    ["title", "goal", "icon"].forEach(k => { if (!L[k]) err(W, "missing", k); });
    if (L.boss && i !== LV.length - 1) err(W, "boss level must be last");
    const html = (L.lesson || {}).html || "";
    if (html.length < 600) err(W, `lesson.html too short (${html.length} chars)`);
    if (/\$\$|\\frac|\\sum|\\sqrt/.test(html)) err(W, "lesson has LaTeX: write math with Unicode (√ Σ ² ·) or .lesson-formula");
    const opens = (html.match(/<(div|table|ul|ol|p|h3|h4|b|strong|code|span|li|tr|td|th|thead|tbody)\b/g) || []).length;
    const closes = (html.match(/<\/(div|table|ul|ol|p|h3|h4|b|strong|code|span|li|tr|td|th|thead|tbody)>/g) || []).length;
    if (opens !== closes) err(W, `lesson html tag balance off: ${opens} opens vs ${closes} closes`);
    const lref = refsIn(html);
    lref.forEach(id => { cited.add(id); if (!byId.has(id)) err(W, "lesson cites unknown question", id); });
    if (lref.length < 2) err(W, "lesson cites fewer than 2 exam questions (ground it in the exams)");
    const qs = L.questions || [];
    if (qs.length < 5 || qs.length > 12) err(W, `has ${qs.length} questions (want 5–12)`);
    qs.forEach((q, j) => {
      total++;
      if (q.ref) {
        totalRefs++;
        const b = byId.get(q.ref), w = `${W} ref ${q.ref}`;
        if (!b) return err(w, "unknown bank question");
        if (!COURSE_TOPICS[cid].includes(b.topic)) warn(w, `bank topic ${b.topic} is outside this course`);
        if (!q.recap) err(w, "missing recap");
        if (qIds.has(`${L.id}:${q.ref}`)) err(w, "same exam question twice in a level"); qIds.add(`${L.id}:${q.ref}`);
        cited.add(q.ref);
        refsIn(q.explanation).forEach(id => { if (!byId.has(id)) err(w, "explanation cites unknown", id); });
        return;
      }
      const w = `${W} ${q.id || "#" + (j + 1)}`;
      if (!q.id || !q.id.startsWith(L.id + "-Q")) err(w, `question id must start with ${L.id}-Q`);
      if (qIds.has(q.id)) err(w, "duplicate question id"); qIds.add(q.id);
      if (!q.question) err(w, "missing question");
      if (!q.recap) err(w, "missing recap");
      if (!q.explanation || q.explanation.length < 40) err(w, "missing/short explanation");
      const o = q.options || [];
      if (o.length < 3 || o.length > 6) err(w, `${o.length} options (want 3–6)`);
      if (!Number.isInteger(q.answer) || q.answer < 0 || q.answer >= o.length) err(w, "answer index out of range:", q.answer);
      o.forEach((v, k) => { if (typeof v !== "string" || !v.trim()) err(w, "empty option", k); });
      if (new Set(o.map(v => String(v).trim())).size !== o.length) err(w, "two options have identical text");
      if (o.some(v => /^(תשובות?|אפשרויות?) [א-ו]( |$)|[א-ו] ו-?[א-ו] נכונות/.test(v))) err(w, "option cites siblings by letter (options are shuffled)");
      if (refsIn(q.question).length) err(w, "question stem cites an exam id (the peek would reveal the answer)");
      refsIn(q.explanation).forEach(id => { cited.add(id); if (!byId.has(id)) err(w, "explanation cites unknown", id); });
      if (q.image && !fs.existsSync(path.join(ROOT, q.image))) err(w, "image missing", q.image);
      if (q.table) {
        const t = q.table, n = (t.head || t.rows[0] || []).length;
        if (!Array.isArray(t.rows) || !t.rows.length) err(w, "table has no rows");
        else t.rows.forEach((r, k) => { if (r.length !== n) err(w, `table row ${k} has ${r.length} cells, header ${n}`); });
      }
      ["question", "explanation", "recap"].forEach(k => {
        const s = String(q[k] || "");
        if ((s.match(/\*\*/g) || []).length % 2) err(w, k, "has an unpaired **");
        if ((s.match(/`/g) || []).length % 2) err(w, k, "has an unpaired backtick");
      });
    });
    const nRef = qs.filter(q => q.ref).length;
    if (nRef < 1) err(W, "no real exam question (ref) in this level");
  });
});

// index.html must load every level file, exactly once
const IDX = path.join(ROOT, "index.html");
let html = fs.readFileSync(IDX, "utf8");
const block = files.map(f => `<script src="${f}?v=1"></script>`).join("\n");
const re = /(<!-- COURSE LEVELS -->\n)([\s\S]*?)(<!-- \/COURSE LEVELS -->)/;
if (!re.test(html)) err("index.html has no COURSE LEVELS block");
else if (args.includes("--sync")) {
  html = html.replace(re, (m, a, b, c) => a + (block ? block + "\n" : "") + c);
  fs.writeFileSync(IDX, html);
  console.log(`index.html: ${files.length} level scripts`);
} else {
  const listed = [...html.match(re)[2].matchAll(/src="([^"?]+)/g)].map(m => m[1]);
  files.forEach(f => { if (!listed.includes(f)) err("index.html does not load", f, "(run with --sync)"); });
  listed.forEach(f => { if (!files.includes(f)) err("index.html loads a missing file", f); });
}

const di = args.indexOf("--dump");
if (di >= 0) {
  const cid = args[di + 1], only = args[di + 2];
  (COURSES[cid] || { levels: [] }).levels.filter(L => !only || L.id === only).forEach(L => {
    console.log(`\n=== ${L.id} ${L.title}`);
    L.questions.forEach(q => {
      if (q.ref) { const b = byId.get(q.ref); console.log(`\n[REF ${q.ref}] ${b.question}\n` + b.options.map((o, k) => `  ${k === b.correctIndex ? "*" : " "} ${o}`).join("\n")); return; }
      console.log(`\n[${q.id}] ${q.question}` + (q.code ? "\n" + q.code : "") + (q.table ? "\n" + JSON.stringify(q.table) : "") + "\n" +
        q.options.map((o, k) => `  ${k === q.answer ? "*" : " "} ${o}`).join("\n") + `\n  → ${q.explanation}`);
    });
  });
}

const per = Object.keys(COURSE_TOPICS).map(c => { const L = (COURSES[c] || { levels: [] }).levels; return `${c}: ${L.length} levels / ${L.reduce((n, l) => n + l.questions.length, 0)} q`; });
console.log(`\n${per.join(" · ")}\ntotal ${total} questions (${totalRefs} real exam questions), ${cited.size} distinct exam ids cited`);
if (bad) { console.log(`\nCOURSE CHECK FAILED: ${bad} error(s)`); process.exit(1); }
console.log("COURSE CHECK PASSED");
