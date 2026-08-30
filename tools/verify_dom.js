// Headless DOM render test (no real browser). Loads index.html + questions.js +
// learn.js + app.js in jsdom and asserts the app boots, renders topics, and can
// start a practice session and render a question with options.
const fs = require("fs"), path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");

const dom = new JSDOM(html, { pretendToBeVisual: true, url: "http://localhost/" });
const { window } = dom;
// stubs for APIs jsdom lacks
window.scrollTo = () => {};
window.matchMedia = window.matchMedia || (q => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }));
window.HTMLElement.prototype.scrollIntoView = () => {};
window.HTMLElement.prototype.focus = window.HTMLElement.prototype.focus || (() => {});
window.confirm = () => true;

global.window = window; global.document = window.document; global.localStorage = window.localStorage;

function run(file) { const code = fs.readFileSync(path.join(ROOT, file), "utf8"); window.eval(code); }
run("questions.js");
run("learn.js");
run("app.js");

const $ = s => window.document.querySelector(s);
let fail = 0;
function assert(cond, msg) { console.log((cond ? "  ok  " : "FAIL  ") + msg); if (!cond) fail++; }

assert(Array.isArray(window.QUESTIONS) && window.QUESTIONS.length > 200, `questions loaded: ${window.QUESTIONS.length}`);
assert(Array.isArray(window.LEARN) && window.LEARN.length === 14, `learn chapters: ${window.LEARN.length}`);

// ---- exam briefs: every chapter renders, and its cross-references resolve ----
const LEARN = window.LEARN;
const briefHtml = LEARN.map(c => c.html).join("");
const qIds = new Set(window.QUESTIONS.map(q => q.id));
const topicKeys = new Set(window.QUESTIONS.map(q => q.topic));

assert(LEARN.every(c => c.html && c.html.length > 500), "every chapter has content");
assert(LEARN.every(c => topicKeys.has(c.topic)), "every chapter maps to a real topic key");
assert(!/@@MATH\d+@@/.test(briefHtml), "no unrendered math placeholders left");
assert(!/math-fail/.test(briefHtml), "no math failed to render");
assert(!/\[\[/.test(briefHtml), "no unresolved [[wikilinks]] left");

const mathNodes = briefHtml.match(/<math /g) || [];
const ltrMath = briefHtml.match(/<math dir="ltr"/g) || [];
assert(mathNodes.length > 700, `MathML formulas rendered: ${mathNodes.length}`);
assert(mathNodes.length === ltrMath.length, `all math is dir="ltr" (RTL page would scramble it)`);

// a dangling question reference would open an empty peek panel
const refs = [...briefHtml.matchAll(/class="qref" data-q="([^"]+)"/g)].map(m => m[1]);
const dangling = [...new Set(refs)].filter(id => !qIds.has(id));
assert(refs.length > 900, `question references linked: ${refs.length}`);
assert(dangling.length === 0, `all question refs resolve${dangling.length ? " DANGLING: " + dangling.slice(0, 3).join(", ") : ""}`);

const chapterIds = new Set(LEARN.map(c => c.id));
const xrefs = [...new Set([...briefHtml.matchAll(/class="xref" data-chapter="([^"]+)"/g)].map(m => m[1]))];
assert(xrefs.length > 0 && xrefs.every(id => chapterIds.has(id)), `all ${xrefs.length} chapter xrefs resolve`);
assert((briefHtml.match(/<table>/g) || []).length === (briefHtml.match(/tbl-wrap/g) || []).length,
  "every table is wrapped for horizontal scroll");
const topicBtns = window.document.querySelectorAll("#topicGrid .topic-btn");
assert(topicBtns.length === 14, `topic buttons rendered: ${topicBtns.length} (13 topics + all)`);
assert(/\d/.test($("#datasetInfo").textContent), `datasetInfo shows count: "${$("#datasetInfo").textContent}"`);
assert($("#topStats").textContent.includes("במאגר"), `top stats populated`);
assert(!$("#learnEntry").style.display || $("#learnEntry").style.display !== "none", `learn entry visible`);

// start a free-practice session and render the first question
$("#startBtn").click();
const quizVisible = !$("#screen-quiz").classList.contains("hidden");
assert(quizVisible, "quiz screen shown after start");
assert($("#questionText").textContent.trim().length > 0, `question text rendered (${$("#questionText").textContent.length} chars)`);
const opts = window.document.querySelectorAll("#optionsList .opt");
assert(opts.length >= 2, `options rendered: ${opts.length}`);

// answer the first option, expect feedback to appear
opts[0].click();
assert(!$("#feedback").classList.contains("hidden"), "feedback shown after answering");

// ---- learn section: open it, render a brief, peek at a cited question ----
$("#quitBtn").click();                       // leave the practice session
$("#learnEntry").click();
assert(!$("#screen-learn").classList.contains("hidden"), "learn screen shown");
const tocBtns = window.document.querySelectorAll("#learnToc button");
assert(tocBtns.length === 14, `TOC entries: ${tocBtns.length}`);
assert(window.document.querySelectorAll("#learnToc button.sub").length === 1,
  "the ID3 companion chapter is nested under Decision Trees");
assert(!!$("#learnContent .brief-meta"), "brief meta line rendered");
assert(!!$("#learnContent .learn-drill"), "drill-this-topic button rendered");
assert($("#learnContent math") !== null || LEARN[0].html.indexOf("<math") === -1,
  "MathML reaches the DOM");

const firstQref = $("#learnContent .qref");
assert(!!firstQref, "question-id chips render inside the brief");
firstQref.click();
const peekOpen = !$("#qpeek").classList.contains("hidden");
assert(peekOpen, "peek panel opens on clicking a question id");
const peekedId = $("#qpeekId").textContent.trim();
const peeked = window.QUESTIONS.find(q => q.id === peekedId);
assert(!!peeked, `peek shows a real question: ${peekedId}`);
assert($("#qpeekBody").textContent.includes(peeked.question.slice(0, 25)),
  "peek body shows that question's text");
assert(window.document.querySelectorAll("#qpeekBody .qpeek-opts li").length === peeked.options.length,
  `peek lists all ${peeked.options.length} options`);
assert(window.document.querySelectorAll("#qpeekBody .qpeek-opts li.right").length === 1,
  "peek marks exactly one option correct");
assert(window.document.querySelectorAll("#qpeekBody .qpeek-opts li")[peeked.correctIndex]
  .classList.contains("right"), "peek marks the RIGHT option correct");
$("#qpeekClose").click();
assert($("#qpeek").classList.contains("hidden"), "peek closes");

// figure integrity: every referenced image file must exist on disk
const imgRefs = [];
for (const q of window.QUESTIONS) {
  if (q.image) imgRefs.push(q.image);
  for (const o of q.options) if (String(o).startsWith("img:")) imgRefs.push(String(o).slice(4));
}
const missingImgs = imgRefs.filter(r => !fs.existsSync(path.join(ROOT, r)));
assert(imgRefs.length >= 30, `figure image references: ${imgRefs.length}`);
assert(missingImgs.length === 0, `all figure images exist on disk${missingImgs.length ? " MISSING: " + missingImgs.slice(0, 3).join(", ") : ""}`);

console.log(fail === 0 ? "\nDOM VERIFY PASSED" : `\nDOM VERIFY FAILED (${fail})`);
process.exit(fail === 0 ? 0 : 1);
