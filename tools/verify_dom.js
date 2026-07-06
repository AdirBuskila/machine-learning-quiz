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
assert(Array.isArray(window.LEARN) && window.LEARN.length === 13, `learn chapters: ${window.LEARN.length}`);
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
