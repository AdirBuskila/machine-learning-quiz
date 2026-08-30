// Drive the learning section (the per-subject exam briefs) in real Chrome and assert
// it renders correctly on desktop AND mobile. jsdom can't do this: MathML layout,
// horizontal overflow and sticky/scroll behaviour only exist in a real engine.
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "file:///" + path.join(__dirname, "..", "index.html").replace(/\\/g, "/");
const SHOT = p => path.join(__dirname, "raw", p);

let failed = 0;
function assert(cond, msg){
  console.log((cond ? "  ok  " : "FAIL  ") + msg);
  if(!cond){ failed++; process.exitCode = 1; }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const b = await puppeteer.launch({ executablePath: CHROME, headless: "new",
    args: ["--allow-file-access-from-files", "--no-sandbox"] });
  const pg = await b.newPage();
  await pg.setViewport({ width: 1100, height: 1400, deviceScaleFactor: 1 });
  const errs = [];
  pg.on("pageerror", e => errs.push(String(e)));
  pg.on("console", m => { if (m.type() === "error") errs.push("console:" + m.text()); });
  await pg.goto(URL, { waitUntil: "networkidle0" });

  // screenshots must never catch the .18s "rise" fade mid-flight, or every review
  // of them shows phantom transparency
  const shoot = async name => { await sleep(320); await pg.screenshot({ path: SHOT(name) }); };
  const overflow = () => pg.evaluate(() => ({
    sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
  const gotoChapter = i => pg.evaluate(i => document.querySelectorAll("#learnToc button")[i].click(), i);

  // ---------------- desktop ----------------
  await pg.click("#learnEntry");
  await pg.waitForSelector("#screen-learn:not(.hidden)");
  await pg.waitForSelector("#learnToc button");

  const tocCount = await pg.$$eval("#learnToc button", els => els.length);
  assert(tocCount === 14, `TOC lists 14 chapters (got ${tocCount})`);
  assert(await pg.$$eval("#learnToc button.sub", els => els.length) === 1,
    "the ID3 companion chapter is nested under Decision Trees");
  assert((await pg.$eval(".learn-h", e => e.textContent.trim())).length > 0, "chapter has a title");
  assert(!!(await pg.$(".brief-meta")), "brief meta strip renders");
  await shoot("brief-desktop-trees.png");

  // MathML must actually lay out — a formula that renders 0px wide is invisible math.
  const knn = await pg.$$eval("#learnToc button", els =>
    els.findIndex(e => e.textContent.includes("KNN")));
  await gotoChapter(knn);
  await pg.waitForSelector("#learnContent math");
  const math = await pg.$$eval("#learnContent math", els => els.map(m => {
    const r = m.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height), dir: m.getAttribute("dir") };
  }));
  assert(math.length > 20, `MathML formulas on the KNN page: ${math.length}`);
  assert(math.every(m => m.w > 0 && m.h > 0), "every formula has real layout size (not invisible)");
  assert(math.every(m => m.dir === "ltr"), "every formula renders LTR inside the RTL page");
  await shoot("brief-desktop-knn-math.png");

  // callouts got their semantic colour classes
  const cals = await pg.$$eval("#learnContent blockquote.cal", els => els.length);
  assert(cals > 0, `callouts rendered on the KNN page: ${cals}`);

  // ---- question peek: the core new interaction ----
  await pg.click("#learnContent .qref");
  await pg.waitForSelector("#qpeek:not(.hidden)");
  const peek = await pg.evaluate(() => ({
    id: document.querySelector("#qpeekId").textContent.trim(),
    opts: document.querySelectorAll("#qpeekBody .qpeek-opts li").length,
    right: document.querySelectorAll("#qpeekBody .qpeek-opts li.right").length,
    text: document.querySelector("#qpeekBody .qpeek-q").textContent.trim().length,
  }));
  assert(peek.opts >= 2 && peek.right === 1,
    `peek shows ${peek.id}: ${peek.opts} options, exactly 1 marked correct`);
  assert(peek.text > 10, "peek shows the question text");
  await shoot("brief-desktop-peek.png");
  await pg.click("#qpeekClose");
  await pg.waitForSelector("#qpeek.hidden");

  // ---- drill button hands off to a real practice session on that topic ----
  await pg.click("#learnContent .learn-drill");
  await pg.waitForSelector("#screen-quiz:not(.hidden)");
  const drilled = await pg.evaluate(() => ({ topic: S.topic, mode: S.mode, n: S.pool.length }));
  assert(drilled.mode === "practice" && drilled.topic === "knn" && drilled.n > 0,
    `drill button started a ${drilled.mode} session on "${drilled.topic}" (${drilled.n} questions)`);
  await pg.click("#quitBtn");
  await pg.waitForSelector("#screen-start:not(.hidden)");

  // light theme, a table-heavy chapter, for visual review
  await pg.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  await pg.click("#learnEntry");
  await pg.waitForSelector("#learnToc button");
  await gotoChapter(2);                       // הערכת מודל — the biggest chapter
  await sleep(150);
  await shoot("brief-desktop-light.png");
  let ov = await overflow();
  assert(ov.sw <= ov.cw + 1, `desktop: no horizontal overflow (${ov.sw} <= ${ov.cw})`);

  // ---------------- mobile ----------------
  await pg.evaluate(() => { document.documentElement.dataset.theme = "dark"; });
  await pg.setViewport({ width: 390, height: 780, deviceScaleFactor: 2 });
  await pg.reload({ waitUntil: "networkidle0" });
  await pg.click("#learnEntry");
  await pg.waitForSelector("#screen-learn:not(.hidden)");

  const tocDisplay = () => pg.$eval("#learnToc", el => getComputedStyle(el).display);
  assert(await tocDisplay() === "none", "mobile: chapter list hidden until toggled");
  ov = await overflow();
  assert(ov.sw <= ov.cw + 1, `mobile: no horizontal overflow (${ov.sw} <= ${ov.cw})`);
  await shoot("brief-mobile-trees.png");

  await pg.click("#learnTocToggle");
  await sleep(200);
  assert(await tocDisplay() !== "none", "mobile: chapter list opens on toggle");
  await shoot("brief-mobile-toc.png");
  await gotoChapter(0);
  await sleep(200);
  assert(await tocDisplay() === "none", "mobile: picking a chapter closes the list");

  // EVERY chapter must fit 390px — one wide table or long formula breaks the whole page
  const bad = [];
  for (let i = 0; i < tocCount; i++) {
    await gotoChapter(i);
    await sleep(120);
    const o = await overflow();
    if (o.sw > o.cw + 1) {
      bad.push(`${i}:${await pg.$eval(".learn-h", e => e.textContent.trim())} (${o.sw}>${o.cw})`);
    }
  }
  assert(bad.length === 0,
    `mobile: all ${tocCount} chapters fit 390px${bad.length ? " OVERFLOW: " + bad.join(", ") : ""}`);

  // wide tables scroll inside their own box rather than stretching the page
  const knnM = await pg.$$eval("#learnToc button", els =>
    els.findIndex(e => e.textContent.includes("KNN")));
  await gotoChapter(knnM);
  await sleep(200);
  const tbl = await pg.$$eval(".tbl-wrap", els => els.map(el => ({
    scrolls: el.scrollWidth > el.clientWidth,
    fits: el.getBoundingClientRect().width <= document.documentElement.clientWidth + 1,
  })));
  assert(tbl.every(t => t.fits), `mobile: all ${tbl.length} table boxes fit the viewport`);
  console.log(`       (${tbl.filter(t => t.scrolls).length} of ${tbl.length} scroll internally)`);

  // display math scrolls internally too
  const mb = await pg.$$eval(".math-block", els => els.map(el =>
    el.getBoundingClientRect().width <= document.documentElement.clientWidth + 1));
  assert(mb.every(Boolean), `mobile: all ${mb.length} display formulas fit the viewport`);
  await shoot("brief-mobile-knn.png");

  // peek as a bottom sheet
  await pg.click("#learnContent .qref");
  await pg.waitForSelector("#qpeek:not(.hidden)");
  await sleep(350);                      // let the slide-up animation settle before measuring
  ov = await overflow();
  assert(ov.sw <= ov.cw + 1, `mobile: peek sheet causes no overflow (${ov.sw} <= ${ov.cw})`);
  const sheet = await pg.$eval(".qpeek-panel", el => {
    const r = el.getBoundingClientRect();
    return { w: Math.round(r.width), bottom: Math.round(r.bottom),
             vh: window.innerHeight, vw: window.innerWidth };
  });
  assert(sheet.w >= sheet.vw - 1 && Math.abs(sheet.bottom - sheet.vh) <= 2,
    `mobile: peek is a full-width bottom sheet (${sheet.w}px wide, bottom at ${sheet.bottom}/${sheet.vh})`);
  await shoot("brief-mobile-peek.png");

  console.log("\nchapters         :", tocCount);
  console.log("KNN formulas     :", math.length, "| callouts:", cals);
  console.log("page errors      :", errs.length ? errs.join("\n") : "none");
  await b.close();
  assert(errs.length === 0, "no page/console errors");
  console.log(failed === 0 ? "\nLEARN VERIFY PASSED" : `\nLEARN VERIFY FAILED (${failed})`);
  process.exit(failed === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
