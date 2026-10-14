"use strict";
/* Guided courses — "צריך עזרה?": level map → lesson → level quiz → summary.
   Levels come from courses/<course>/*.js via courses.js (window.COURSES[id].levels). The quiz runs
   on the main engine in app.js (S, renderQuestion, choose, next) with S.mode = "course"; app.js
   calls back into window.COURSE.record / finishLevel / exit. Each course keeps its progress in its
   own localStorage key, apart from the bank stats.

   level    = {id, order, icon, title, goal, boss?, lesson:{html}, questions:[…]}
   question = {id, question, code?, image?, table?, options:[string], answer:<index>, explanation, recap}
            | {ref:"<bank question id>", recap, explanation?}   ← a real exam question, verbatim
   Text in questions/explanations takes **bold**, `code` and newlines; an exam question id
   (e.g. 23S-A-Q3) anywhere in a lesson or explanation becomes a button that opens that question. */
(function(){
  const META = {
    cls:   { name: "מסווגים", title: "צריך עזרה ב-KNN, עצים ובייס?", icon: "🌳", store: "mlq_crs_cls_v1",
             topics: ["knn", "trees", "naive_bayes"],
             intro: "שלושת המסווגים הקלאסיים של הקורס, מאפס ועד שאלת חישוב בגודל של מבחן: פונקציות מרחק וחיזוי KNN, אנטרופיה ו-Information Gain בעץ החלטה, וחישוב נאיב בייס מטבלה. כל שלב פותח בשיעור קצר שבנוי על המבחנים הקודמים וממשיך בשאלות בסגנון המבחן. כל השלבים פתוחים; שלב נחשב עבור מ-80%." },
    eval:  { name: "הערכת מודל", title: "צריך עזרה בהערכת מודל ומושגי יסוד?", icon: "📊", store: "mlq_crs_eval_v1",
             topics: ["intro", "evaluation"],
             intro: "הנושא עם הכי הרבה שאלות במבחן: סוגי למידה, train/test, overfitting, מטריצת בלבול, Precision/Recall/F1, ROC, Cross-Validation ו-Bias/Variance. כל שלב פותח בשיעור קצר שבנוי על המבחנים הקודמים וממשיך בשאלות בסגנון המבחן. כל השלבים פתוחים; שלב נחשב עבור מ-80%." },
    reg:   { name: "רגרסיה ורשתות", title: "צריך עזרה ברגרסיה ורשתות נוירונים?", icon: "🧠", store: "mlq_crs_reg_v1",
             topics: ["regression", "neural_nets", "svm"],
             intro: "מקו ישר ועד רשת נוירונים: רגרסיה לינארית, פונקציית מחיר, Gradient Descent וקצב למידה, רגרסיה לוגיסטית, פרספטרון, פונקציות אקטיבציה, Backpropagation ו-SVM. כל שלב פותח בשיעור קצר שבנוי על המבחנים הקודמים וממשיך בשאלות בסגנון המבחן. כל השלבים פתוחים; שלב נחשב עבור מ-80%." },
    unsup: { name: "אשכול, טקסט ותמונה", title: "צריך עזרה באשכול, טקסט ותמונה?", icon: "🧩", store: "mlq_crs_unsup_v1",
             topics: ["clustering", "text", "image", "ensemble", "flow"],
             intro: "הנושאים הקטנים שביחד שווים הרבה נקודות: K-Means ואשכול היררכי, ניתוח טקסט (BoW, TF-IDF), עיבוד תמונה (פיקסלים, קונבולוציה), למידת אנסמבל ו-Flow. כל שלב פותח בשיעור קצר שבנוי על המבחנים הקודמים וממשיך בשאלות בסגנון המבחן. כל השלבים פתוחים; שלב נחשב עבור מ-80%." },
  };
  const PASS = 80;
  const ALL = window.COURSES || {};
  const BANK = Array.isArray(window.QUESTIONS) ? window.QUESTIONS : [];
  const screen = document.getElementById("screen-course");
  if (!screen) return;
  const has = id => ALL[id] && ALL[id].levels && ALL[id].levels.length;
  Object.keys(META).forEach(id => { if (!has(id)) document.querySelectorAll(`[data-course="${id}"]`).forEach(el => el.style.display = "none"); });
  if (!Object.keys(META).some(has)) { const h = document.getElementById("crsHub"); if (h) h.style.display = "none"; }

  let C = null;            // the open course: {id, meta, levels, prog}
  let cur = null, run = null;   // the level being played, and its answers {qid: ok}
  const prog = {};
  function load(id){ try { return JSON.parse(localStorage.getItem(META[id].store)) || { levels:{} }; } catch(e){ return { levels:{} }; } }
  function save(){ try { localStorage.setItem(C.meta.store, JSON.stringify(C.prog)); } catch(e){} }
  function course(id){ prog[id] = prog[id] || load(id); return { id, meta: META[id], levels: ALL[id].levels, prog: prog[id] }; }

  const views = ["crsMapView", "crsLessonView", "crsSumView"];
  function view(id){ show("screen-course"); views.forEach(v => document.getElementById(v).classList.toggle("hidden", v !== id)); }
  const passedIn = (c, L) => ((c.prog.levels[L.id] || {}).best || 0) >= PASS;
  const passed = L => passedIn(C, L);
  const recommendedIn = c => c.levels.find(L => !passedIn(c, L));
  const levelName = L => `שלב ${L.order} · ${L.title}`;
  const backToMode = () => { S.mode = (document.querySelector('input[name=mode]:checked') || {}).value || "practice"; };

  /* ---------- rendering helpers ---------- */
  const QID = /\b((?:\d{2}[ABS]-[ABC])|SAMP-\d|PRAC-[NX])-Q(\d+)\b/g;
  const bankIds = new Set(BANK.map(q => q.id));
  // exam question ids → peek buttons (only ids that exist in the bank)
  function linkRefs(html){
    return html.replace(QID, m => bankIds.has(m) ? `<button type="button" class="qref" data-q="${m}">${m}</button>` : m);
  }
  /* Bidi: the page is RTL, so a formula like "(2, 5) ו-(7, 1)" or "|7−2| = 5" inside Hebrew gets
     its parentheses and operands reordered. Every maximal non-Hebrew run that holds a digit or a
     Latin letter is wrapped in an LTR isolate. A run may span inline tags (<b>3</b>) as long as
     they open and close inside it; text inside code / formulas / existing isolates is left alone. */
  const INLINE = /^<\/?(b|strong|i|em|u|sub|sup|code)\b[^>]*>$/i;
  const SKIP = /^<(code|bdi|pre)\b|class="[^"]*\b(lesson-formula|qref)\b|\bdir="ltr"/i;
  const START = /[A-Za-z0-9(\[|√‖Σ∑−]/, END = /[A-Za-z0-9)\]|²³‖%∞'"]/;
  function isolate(html){
    const toks = String(html).match(/<[^>]+>|[^<]+/g) || [];
    let out = "", run = [], skip = 0;
    const flush = () => {
      if (!run.length) return;
      const s = run.join("");
      // trim to the first/last "real" character, keeping whole tags; then require balanced tags
      const parts = s.match(/<[^>]+>|[^<]/g);
      let a = 0, b = parts.length - 1;
      while (a <= b && !(parts[a].length === 1 && START.test(parts[a]))) a++;
      while (b >= a && !(parts[b].length === 1 && END.test(parts[b]))) b--;
      // pull in tags hugging the core: <b>3</b> → keep both
      while (a > 0 && parts[a - 1].length > 1 && !/^<\//.test(parts[a - 1])) a--;
      while (b < parts.length - 1 && /^<\//.test(parts[b + 1])) b++;
      const core = parts.slice(a, b + 1), st = [];
      let ok = a <= b && /[A-Za-z0-9]/.test(core.join(""));
      for (const t of core) { if (t.length === 1 || !ok) continue;
        const m = t.match(/^<(\/?)(\w+)/); if (m[1]) { if (st.pop() !== m[2].toLowerCase()) ok = false; } else st.push(m[2].toLowerCase()); }
      if (ok && st.length) ok = false;
      // a multi-word Latin phrase with no math is fine on its own; isolate when it mixes symbols/digits
      out += ok ? parts.slice(0, a).join("") + `<bdi class="ltr${core.join("").replace(/<[^>]+>/g, "").length <= 22 ? " nw" : ""}" dir="ltr">${core.join("")}</bdi>` + parts.slice(b + 1).join("") : s;
      run = [];
    };
    for (const t of toks) {
      if (t[0] === "<") {
        if (skip) { out += t; if (/^<\//.test(t)) skip--; else if (!/\/>$|^<(br|img|hr|wbr)\b/i.test(t)) skip++; continue; }
        if (SKIP.test(t)) { flush(); out += t; if (!/^<(br|img)\b/i.test(t)) skip = 1; continue; }
        if (INLINE.test(t)) { run.push(t); continue; }
        flush(); out += t; continue;
      }
      if (skip) { out += t; continue; }
      // split text at Hebrew letters: Hebrew text ends a run
      for (const piece of t.match(/[֐-׿]+|[^֐-׿]+/g)) {
        if (/[֐-׿]/.test(piece)) { flush(); out += piece; } else run.push(piece);
      }
    }
    flush();
    return out;
  }
  window.crsIsolate = isolate;
  // question / explanation text: escape, then **bold**, `code`, newlines, bidi isolates, exam-id links
  function rich(s){
    let h = escapeHtml(String(s == null ? "" : s));
    h = h.replace(/`([^`]+)`/g, (m, c) => `<code dir="ltr">${c}</code>`)
         .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
         .replace(/\n/g, "<br>");
    return linkRefs(isolate(h));
  }
  window.crsRich = rich;
  // a data table attached to a question: {caption?, head:[…], rows:[[…]]}
  function tableHtml(t){
    if (!t) return "";
    const cell = c => { const s = String(c); return /[֐-׿]/.test(s) ? rich(s) : `<span dir="ltr">${rich(s)}</span>`; };
    return `<div class="tbl-wrap q-table"><table>` +
      (t.head ? `<thead><tr>${t.head.map(h => `<th>${cell(h)}</th>`).join("")}</tr></thead>` : "") +
      `<tbody>${t.rows.map(r => `<tr>${r.map(c => `<td>${cell(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` +
      (t.caption ? `<div class="q-table-cap">${rich(t.caption)}</div>` : "");
  }
  window.crsTable = tableHtml;
  // lesson html: wrap tables for sideways scroll on phones, link exam ids
  function lessonHtml(html){
    return linkRefs(isolate(String(html || "").replace(/<table>/g, '<div class="tbl-wrap"><table>').replace(/<\/table>/g, "</table></div>")));
  }

  /* ---------- map ---------- */
  function renderMap(){
    const LV = C.levels, rec = recommendedIn(C), nPass = LV.filter(passed).length;
    $("#crsTitle").textContent = `${C.meta.icon} ${C.meta.title}`;
    $("#crsIntro").textContent = C.meta.intro;
    $("#crsOverall").innerHTML =
      `עברת <b>${nPass}</b> מתוך <b>${LV.length}</b> שלבים` +
      `<div class="progress" style="margin-top:.5rem"><div class="progress-fill" style="width:${nPass / LV.length * 100}%"></div></div>` +
      (nPass === LV.length ? `<p style="margin:.6rem 0 0">🏆 סיימת את כל הקורס. הצעד הבא: שאלות ${C.meta.name} מהמבחנים.</p>` : "");
    $("#crsMap").innerHTML = LV.map(L => {
      const st = C.prog.levels[L.id];
      const cls = ["crs-lvl", passed(L) ? "passed" : "", L === rec ? "next" : "", L.boss ? "boss" : ""].join(" ");
      const status = !st ? `עוד לא התחלת · ${L.questions.length} שאלות`
        : passed(L) ? `✓ עברת · שיא ${st.best}%` : `ניסיון אחרון ${st.last}% · שיא ${st.best}% (צריך 80%)`;
      return `<button class="${cls}" data-id="${L.id}" type="button">
        <span class="num">שלב ${L.order}${L.boss ? " · מבחן גמר" : ""}</span>
        <span class="ttl">${L.icon || ""} ${escapeHtml(L.title)}</span>
        <span class="gl">${escapeHtml(L.goal || "")}</span>
        <span class="st">${status}</span></button>`;
    }).join("");
    $("#crsMap").querySelectorAll(".crs-lvl").forEach(b => b.onclick = () => openLesson(LV.find(L => L.id === b.dataset.id)));
  }
  function openMap(id){ if (id) C = course(id); cur = null; renderMap(); view("crsMapView"); }

  /* ---------- lesson ---------- */
  function openLesson(L){
    cur = L;
    const nReal = L.questions.filter(q => q.ref).length;
    $("#crsLesson").innerHTML =
      `<div class="q-head"><span class="chip">שלב ${L.order} מתוך ${C.levels.length}</span><span class="chip soft">${L.questions.length} שאלות${nReal ? ` · ${nReal} מהמבחנים` : ""}</span></div>` +
      `<h2 class="learn-h">${L.icon || ""} ${escapeHtml(L.title)}</h2><p class="sub crs-goal">${escapeHtml(L.goal || "")}</p>` +
      `<div class="lesson-body">${lessonHtml(L.lesson && L.lesson.html)}</div>`;
    view("crsLessonView");
  }

  /* ---------- level quiz (runs on the main engine) ---------- */
  // a level question in engine form: {options, correctIndex} like the bank; refs are cloned from the bank
  function engineQ(q, L){
    const levelTitle = levelName(L), badge = `${C.meta.icon} ${C.meta.name}`;
    let base;
    if (q.ref){
      const b = BANK.find(x => x.id === q.ref);
      base = Object.assign({}, b, { id: `${L.id}:${q.ref}`, explanation: q.explanation || b.explanation, recap: q.recap,
        fromExam: b.sourceLabel ? `${b.sourceLabel}, שאלה ${(String(b.id).match(/-Q(\d+)$/) || [])[1] || ""}` : b.id, examId: b.id });
    } else {
      base = Object.assign({}, q, { correctIndex: q.answer });
    }
    return Object.assign(base, { course: { courseId: C.id, levelId: L.id, levelTitle, badge } });
  }
  function startLevel(L){
    cur = L; run = {};
    S.mode = "course"; S._views = null;
    S.pool = L.questions.map(q => engineQ(q, L));
    S.pos = 0; S.hist = []; S.hpos = 0; S.streak = 0;
    $("#prevBtn").classList.remove("hidden");
    show("screen-quiz");
    renderQuestion();
  }
  function record(q, ok){ if (run && !(q.id in run)) run[q.id] = ok; }

  function finishLevel(){
    const L = cur; if (!L) return openMap();
    const qs = S.pool, total = qs.length;
    const correct = qs.filter(q => run && run[q.id]).length;
    const pct = Math.round(correct / total * 100);
    const prev = C.prog.levels[L.id] || {};
    C.prog.levels[L.id] = { best: Math.max(prev.best || 0, pct), last: pct, at: Date.now() };
    save();
    const ok = pct >= PASS, LV = C.levels, nextL = LV[LV.indexOf(L) + 1];
    const wrong = qs.filter(q => !(run && run[q.id]));
    backToMode();
    $("#crsSum").innerHTML =
      `<div class="q-head" style="justify-content:center"><span class="chip">${escapeHtml(levelName(L))}</span></div>` +
      `<div class="big">${pct}%</div><div><b>${correct} / ${total}</b> תשובות נכונות</div>` +
      `<p class="msg">${ok ? (L.boss ? `🏆 עברת את מבחן הגמר של ${C.meta.name}!` : "✓ עברת את השלב! אפשר להתקדם.") :
        "עוד לא 80%. כדאי לקרוא שוב את השיעור ולנסות שוב, זה בדיוק השלב שבו זה נבנה."}</p>` +
      (wrong.length ? `<div class="crs-review"><div class="crs-review-t">כדאי לחזור על:</div>${
        wrong.map(q => `<div>• ${rich(q.recap || q.question)}</div>`).join("")}</div>` : "") +
      `<div class="crs-actions">` +
        (nextL ? `<button class="btn ${ok ? "primary" : "ghost"}" data-act="next">לשלב הבא: ${escapeHtml(nextL.title)} ⟵</button>` : "") +
        (L.boss || !nextL ? `<button class="btn primary" data-act="real">לשאלות ${C.meta.name} מהמבחנים האמיתיים ⟵</button>` : "") +
        `<button class="btn ${ok ? "ghost" : "primary"}" data-act="retry">🔁 לנסות שוב</button>` +
        `<button class="btn ghost" data-act="lesson">📘 לשיעור</button>` +
        `<button class="btn ghost" data-act="map">🗺️ למפת השלבים</button>` +
      `</div>`;
    $("#crsSum").querySelectorAll("[data-act]").forEach(b => b.onclick = () => {
      const a = b.dataset.act;
      if (a === "next") openLesson(nextL);
      else if (a === "retry") startLevel(L);
      else if (a === "lesson") openLesson(L);
      else if (a === "real") realExam();
      else openMap();
    });
    view("crsSumView");
  }
  function exit(){ backToMode(); S.hist = []; S.hpos = 0; cur ? openLesson(cur) : openMap(); }

  // hand off to the real exam questions of this course's topics (free practice)
  function realExam(){
    const r = document.querySelector('input[name=mode][value="practice"]'); if (r) r.checked = true;
    $("#examOpts").classList.add("hidden");
    ["#officialOnly", "#mistakesOnly"].forEach(s => { const el = $(s); if (el) el.checked = false; });
    S.mode = "practice"; S._views = null;
    S.pool = shuffle(dedupePool(BANK.filter(q => C.meta.topics.includes(q.topic))));
    S.pos = 0; S.hist = []; S.hpos = 0; S.streak = 0;
    $("#prevBtn").classList.remove("hidden");
    show("screen-quiz");
    renderQuestion();
  }

  /* ---------- wiring ---------- */
  $("#crsLessonGo").onclick = () => cur && startLevel(cur);
  $("#crsLessonBack").onclick = () => openMap();
  $("#crsHomeBtn").onclick = () => quit();
  $("#crsResetBtn").onclick = () => { if (confirm(`לאפס את ההתקדמות בקורס ${C.meta.name}?`)){ C.prog.levels = {}; save(); renderMap(); } };
  function paintPromos(){
    document.querySelectorAll(".crs-promo[data-course]").forEach(el => {
      const id = el.dataset.course; if (!has(id)) return;
      const c = course(id), n = c.levels.filter(L => passedIn(c, L)).length, rec = recommendedIn(c);
      const sub = el.querySelector(".crs-promo-sub");
      if (!sub.dataset.base) sub.dataset.base = sub.textContent;
      sub.textContent = n ? `עברת ${n} מתוך ${c.levels.length} שלבים${rec ? ". הבא בתור: " + rec.title : ". סיימת, כל הכבוד!"}` : sub.dataset.base;
    });
  }
  document.querySelectorAll("[data-course]").forEach(el => {
    const id = el.dataset.course; if (!has(id)) return;
    el.onclick = () => openMap(id);
    if (el.classList.contains("crs-promo"))
      el.onkeydown = e => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); openMap(id); } };
  });
  paintPromos();
  window.addEventListener("ml:home", paintPromos);

  // exam-id buttons inside a lesson, the answer feedback or the summary open the question peek
  document.addEventListener("click", e => {
    const b = e.target.closest(".qref"); if (!b || b.closest("#learnContent")) return;
    if (window.openQPeek) window.openQPeek(b.dataset.q);
  });

  // header: one entry that brings you to the course cards on the home screen
  const hub = $("#crsHub");
  if (hub) hub.onclick = () => { quit(); setTimeout(() => { const p = $(".crs-promos"); if (p) p.scrollIntoView({ behavior: "smooth", block: "start" }); }, 60); };

  window.COURSE = { record, finishLevel, exit, openMap, openLesson: L => openLesson(L), startLevel: L => startLevel(L), META };

  // deep link: ?course=cls (map) · ?course=cls&level=C03 (that lesson) · …&play=1 (its quiz)
  try {
    const p = new URLSearchParams(location.search), id = p.get("course");
    if (id && has(id)){
      C = course(id);
      const L = C.levels.find(l => l.id === p.get("level"));
      if (!L) openMap(); else if (p.get("play")) startLevel(L); else openLesson(L);
    }
  } catch(e){}
})();
