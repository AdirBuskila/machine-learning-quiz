# -*- coding: utf-8 -*-
"""E08 · סילום, סטיית תקן וחריגים — video script (build: tools/video/vid.py E08).

Source: courses/eval/E08.js (goal: which algorithms need scaling, what min-max and z-score do, the mean and the
sample standard deviation (n-1), and deciding whether a value is an outlier). Exam anchor: SAMP-1-Q2 (three equally
spaced values, min-max to [-1, 1], a new value equal to the old mean, scale again), which no guided quiz uses; no quiz
item repeats it.
Not used on purpose: the E08 quiz data (2/4/6/8, 10/14/18/22 + 30 with threshold 3, 10/20/25/50, the z-score
statements, multiplying every vector by 3) and the refs SAMP-2-Q24 (+ its twin SAMP-3-Q24), SAMP-1-Q5, SAMP-1-Q20;
PRAC-X-Q30 and PRAC-X-Q25, which mirror quiz items.
Worked examples: three people (age, salary) whose nearest neighbour flips after min-max; the values 3/7/9/13
(min-max, mean 8, sample σ = √(52/3) ≈ 4.16 vs √13 ≈ 3.61 with n) and the value 16 against a 2σ threshold, where the
verdict depends on n-1. Everything asserted with numpy / sklearn.
"""
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

# ---------------------------------------------------------------- nearest neighbour before / after min-max (verified)
PEOPLE = {"אור": (30, 12000), "בן": (55, 12100), "גל": (31, 13000)}
RANGE = ((20, 60), (5000, 20000))
d = lambda p, q: math.dist(p, q)
raw_ab, raw_ac = d(PEOPLE["אור"], PEOPLE["בן"]), d(PEOPLE["אור"], PEOPLE["גל"])
assert raw_ab < raw_ac and round(raw_ab) == 103 and round(raw_ac) == 1000            # raw: B is nearest
mm = lambda v: tuple((x - lo) / (hi - lo) for x, (lo, hi) in zip(v, RANGE))
S = {k: mm(v) for k, v in PEOPLE.items()}
sc_ab, sc_ac = d(S["אור"], S["בן"]), d(S["אור"], S["גל"])
assert sc_ac < sc_ab and f"{sc_ab:.2f}" == "0.63" and f"{sc_ac:.2f}" == "0.07"          # scaled: C is nearest
assert (15000 - 5000) / 15000 == 2 / 3                                                 # a tree split, same split

# ---------------------------------------------------------------- 3 / 7 / 9 / 13 (verified)
V = np.array([3.0, 7.0, 9.0, 13.0])
MM = MinMaxScaler().fit_transform(V.reshape(-1, 1)).ravel()
assert np.allclose(MM, [0, 0.4, 0.6, 1])
assert V.mean() == 8 and ((V - 8) ** 2).sum() == 52
SD1, SD0 = math.sqrt(52 / 3), math.sqrt(13)
assert abs(SD1 - V.std(ddof=1)) < 1e-12 and abs(SD0 - V.std(ddof=0)) < 1e-12
assert f"{SD1:.2f}" == "4.16" and f"{SD0:.2f}" == "3.61"
Z = (V - 8) / SD1
assert [f"{z:.2f}" for z in Z] == ["-1.20", "-0.24", "0.24", "1.20"] and abs(Z.mean()) < 1e-12
assert abs(Z.std(ddof=1) - 1) < 1e-12
assert np.allclose(StandardScaler().fit_transform(V.reshape(-1, 1)).ravel(), (V - 8) / SD0)   # sklearn divides by n
NEW, THR = 16.0, 2
k1, k0 = (NEW - 8) / SD1, (NEW - 8) / SD0
assert f"{k1:.2f}" == "1.92" and k1 < THR and f"{k0:.2f}" == "2.22" and k0 > THR      # the verdict flips with n

# ---------------------------------------------------------------- exam SAMP-1-Q2 (verified on 10 / 20 / 30)
Q = BANK["SAMP-1-Q2"]
assert Q["correctIndex"] == 3 and "[-1,1]" in Q["question"]
E = np.array([10.0, 20.0, 30.0])
sc = MinMaxScaler(feature_range=(-1, 1))
E1 = sc.fit_transform(E.reshape(-1, 1)).ravel()
E2 = sc.fit_transform(np.r_[E, E.mean()].reshape(-1, 1)).ravel()
assert np.allclose(E1, [-1, 0, 1]) and np.allclose(E2, [-1, 0, 1, 0])                # nothing moved
assert E.std(ddof=1) == 10 and abs(np.r_[E, 20].std(ddof=1) - math.sqrt(200 / 3)) < 1e-12   # σ shrinks: 10 -> 8.16
assert np.r_[E, 20].std(ddof=1) < E.std(ddof=1)
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
PT = w.table("pp", ["", "גיל", "שכר (₪)"],
             [[k, str(v[0]), f"{v[1]:,}"] for k, v in PEOPLE.items()], hide_rows=False, size=30)
L = lambda t: f"<span class='ev-l'>{t}</span>"
VT = w.table("vtt", ["$x$", "min-max", "$x - \\bar{x}$", "$(x - \\bar{x})^2$", "z"],
             [[L(int(x)), L(f"{m:g}"), L(int(x - 8)), L(int((x - 8) ** 2)), L(f"{z:.2f}")] for x, m, z in zip(V, MM, Z)],
             hide_rows=False, size=28)
NL = w.Plot("nl", (0, 18), (0, 1), w=900, h=150, xticks=(0, 4, 8, 12, 16), yticks=(), pad=(30, 30, 20, 50), grid=False)
NL.band("b2", 8 - 2 * SD1, 8 + 2 * SD1, "ok")
NL.pts("v", [(x, 0.45) for x in V], "a", r=12)
NL.pts("m", [(8, 0.45)], "ink", r=7)
NL.pts("new", [(NEW, 0.45)], "b", r=13, hidden=True)
NL.text("lm", 8, 0.95, "ממוצע 8", "ink", size=22, hidden=False)
NLSVG = NL.svg()

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבין למה צריך סילום, ולמי, ונכיר שתי שיטות: min-max ו-z-score.",
      say="בסרטון הזה נבין למה צריך סילום, ולמי, ונכיר שתי שיטות: מִין מַקְס, וזֶד סְקוֹר."),
    P("נחשב ממוצע וסטיית תקן כמו שהקורס דורש, ונחליט אם ערך חדש הוא חריג.", hold=0.4),
], stage='<div class="kicker">E08 · הערכת מודל</div><h1>סילום, סטיית תקן וחריגים</h1>'
         '<p>מי צריך סילום, מה עושים min-max ו-z-score, סטיית תקן עם n−1, וזיהוי ערך חריג.</p>', layout="title")

why = Scene("why", "למה סילום", [
    P("שלושה אנשים: גיל ושכר. מי השכן הכי קרוב של אור?", do=show("pp")),
    P("לפי המרחק הרגיל, השכן הוא בן: מרחק של בערך 103. אבל בן מבוגר מאור ב-25 שנה!", do=show("y-1"),
      say="לפי המרחק הרגיל, השכן הוא בן: מרחק של בערך מאה ושלוש. אבל בן מבוגר מאור בעשרים וחמש שנה!"),
    P("גל כמעט באותו גיל, אבל השכר שלו שונה באלף שקל. ואלף שקל ״שוקלים״ יותר מעשרים וחמש שנה.", do=show("y-2")),
    P("המאפיין עם המספרים הגדולים משתלט על המרחק. סילום מביא את שניהם לאותו טווח.", do=show("y-k")),
    P("בנתוני האימון הגיל נע בין 20 ל-60, והשכר בין 5,000 ל-20,000. כל מאפיין מכווצים לפי הטווח שלו.", do=show("y-r"),
      say="בנתוני האימון הגיל נע בין עשרים לשישים, והשכר בין חמשת אלפים לעשרים אלף. כל מאפיין מכווצים לפי הטווח שלו."),
    P("אחרי min-max לטווח אפס עד אחת, המרחקים הם 0.63 ו-0.07. עכשיו השכן הקרוב הוא גל.", do=show("y-3"),
      say="אחרי מִין מַקְס לטווח אפס עד אחת, המרחקים הם אפס נקודה שישים ושלוש, ואפס נקודה אפס שבע. עכשיו השכן הקרוב הוא גל."),
    P("לכן כל מה שמחשב מרחק או סכום משוקלל של מאפיינים צריך סילום.", do=show("y-4")),
    P("ועץ החלטה? הפיצול ״שכר מעל 15,000״ הופך ל״שכר מעל שני שלישים״. אותן דוגמאות בכל צד, אותו עץ.", do=show("y-5"),
      say="ועץ החלטה? הפיצול שכר מעל חמש עשרה אלף, הופך לשכר מעל שני שלישים. אותן דוגמאות בכל צד, אותו עץ.", hold=0.4),
], stage=w.col(PT, w.note("טווחי האימון: גיל 20 עד 60 · שכר 5,000 עד 20,000", "y-r", hidden=True),
               st.card("y-k", "rule", "🔑 בלי סילום, המאפיין עם הטווח הגדול <b>שולט</b> במרחק"), gap=24),
   panel=st.card("y-1", "ex", "גולמי: אור–בן $\\approx 103$")
         + st.card("y-2", "ex", "גולמי: אור–גל $\\approx 1000$")
         + st.card("y-3", "ex", "אחרי סילום: אור–בן $\\approx 0.63$ · אור–גל $\\approx 0.07$")
         + st.card("y-4", "plain", "מרחק / סכום משוקלל ← צריך סילום")
         + st.card("y-5", "plain", "סף על מאפיין <b>אחד</b> ← הסדר נשמר"))

two = Scene("two", "שתי שיטות", [
    P("שתי שיטות. נבדוק אותן על הערכים שלוש, שבע, תשע ושלוש עשרה.", do=show("vt")),
    P("min-max: מחסירים את המינימום ומחלקים בטווח. כאן מינוס שלוש, חלקי עשר. יוצא אפס, 0.4, 0.6 ואחת.",
      do=show("t-1"),
      say="מִין מַקְס: מחסירים את המינימום ומחלקים בטווח. כאן מינוס שלוש, חלקי עשר. יוצא אפס, אפס נקודה ארבע, אפס נקודה שש, ואחת."),
    P("השיטה רואה רק את שני הקצוות, ולא משנה את צורת הפיזור.", hold=0.3),
    P("z-score: מחסירים את הממוצע ומחלקים בסטיית התקן. הממוצע: 32 חלקי 4, כלומר 8.", do=show("t-2"),
      say="זֶד סְקוֹר: מחסירים את הממוצע ומחלקים בסטיית התקן. הממוצע: שלושים ושתיים חלקי ארבע, כלומר שמונה."),
    P("סכום ריבועי הסטיות: עשרים וחמש, אחת, אחת ועשרים וחמש. ביחד חמישים ושתיים.", do=show("t-3")),
    P("ובקורס הזה מחלקים ב-**n פחות אחת**, שלוש: השורש של 52 חלקי 3, בערך 4.16.", do=show("t-4"),
      say="ובקורס הזה מחלקים באֶן פחות אחת, כלומר בשלוש: השורש של חמישים ושתיים חלקי שלוש, בערך ארבע נקודה שש עשרה."),
    P("חלוקה בארבע הייתה נותנת 3.61. זה ההבדל שמבחנים בודקים.", do=show("t-t"),
      say="חלוקה בארבע הייתה נותנת שלוש נקודה שישים ואחת. זה ההבדל שמבחנים בודקים."),
    P("ובעמודה האחרונה: כל סטייה חלקי 4.16. יוצא מינוס 1.20, מינוס 0.24, 0.24 ו-1.20.", do=show("t-5"),
      say="ובעמודה האחרונה: כל סטייה חלקי ארבע נקודה שש עשרה. יוצא מינוס אחת נקודה עשרים, מינוס אפס נקודה עשרים וארבע, אפס נקודה עשרים וארבע, ואחת נקודה עשרים.", hold=0.4),
], stage=w.col('<div id="vt" class="h">' + VT + "</div>"),
   panel=st.card("t-1", "ex", "$x' = \\dfrac{x - \\min}{\\max - \\min} = \\dfrac{x - 3}{10}$")
         + st.card("t-2", "ex", "$\\bar{x} = \\dfrac{3 + 7 + 9 + 13}{4} = 8$")
         + st.card("t-3", "plain", "$\\sum (x - \\bar{x})^2 = 52$")
         + st.card("t-4", "rule", "🔑 $\\sigma = \\sqrt{\\dfrac{52}{4 - 1}} \\approx 4.16$")
         + st.card("t-t", "trap", "⚠️ $\\sqrt{52 / 4} \\approx 3.61$: חלוקה ב-$n$, לא בקורס")
         + st.card("t-5", "plain", "$z = \\dfrac{x - 8}{4.16}$ (העמודה האחרונה)"))

out = Scene("out", "ערך חריג", [
    P("עכשיו מגיע ערך חדש, 16. האם הוא חריג? נגדיר: חריג הוא מי שרחוק יותר משתי סטיות תקן מהממוצע.",
      do=show("nl-new"),
      say="עכשיו מגיע ערך חדש, שש עשרה. האם הוא חריג? נגדיר: חריג הוא מי שרחוק יותר משתי סטיות תקן מהממוצע."),
    P("מודדים את המרחק ביחידות של סטיות תקן: 16 פחות 8, חלקי 4.16. בערך 1.92.", do=show("o-1", "nl-b2"),
      say="מודדים את המרחק ביחידות של סטיות תקן: שש עשרה פחות שמונה, חלקי ארבע נקודה שש עשרה. בערך אחת נקודה תשעים ושתיים."),
    P("פחות משתיים, ולכן **משאירים** אותו. הוא בתוך הפס הירוק.", do=show("o-2"), hold=0.4),
], stage=w.col(NLSVG, w.note("הפס הירוק: עד שתי סטיות תקן מהממוצע"), gap=10),
   panel=st.card("o-1", "ex", "$\\dfrac{|16 - 8|}{4.16} \\approx 1.92$")
         + st.card("o-2", "rule", "🔑 קטן מהסף ← <b>משאירים</b> · גדול מהסף ← מסננים"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: שלושה ערכים במרווחים שווים. מסלמים ב-min-max לטווח מינוס אחת עד אחת.",
      say="שאלה ממבחן: שלושה ערכים במרווחים שווים. מסלמים במִין מַקְס לטווח מינוס אחת עד אחת."),
    P("אחר כך נוסף ערך ששווה לממוצע המקורי, ומסלמים שוב. מה קורה לערכים?"),
    P("ננסה על 10, 20, 30. הסילום שולח את המינימום למינוס אחת, ואת המקסימום לאחת: מינוס אחת, אפס, אחת.",
      do=show("x-1"),
      say="ננסה על עשר, עשרים, שלושים. הסילום שולח את המינימום למינוס אחת, ואת המקסימום לאחת: מינוס אחת, אפס, אחת."),
    P("הערך החדש הוא הממוצע, 20. הוא באמצע. המינימום עדיין 10, והמקסימום עדיין 30.", do=show("x-2"),
      say="הערך החדש הוא הממוצע, עשרים. הוא באמצע. המינימום עדיין עשר, והמקסימום עדיין שלושים."),
    P("ו-min-max תלוי **רק** במינימום ובמקסימום. לכן הסילום השני נותן בדיוק אותם ערכים: אין שום שינוי.",
      do=show("x-3", "q-ok-3"),
      say="ומִין מַקְס תלוי רק במינימום ובמקסימום. לכן הסילום השני נותן בדיוק אותם ערכים: אין שום שינוי."),
    P("שלוש הטענות האחרות מסבירות בעזרת סטיית התקן. אבל min-max בכלל לא משתמש בה.",
      do=show("q-x-0", "q-n-0", "q-x-1", "q-n-1"),
      say="שלוש הטענות האחרות מסבירות בעזרת סטיית התקן. אבל מִין מַקְס בכלל לא משתמש בה."),
    P("ואחת מהן גם טועה בכיוון: ערך באמצע **מקטין** את סטיית התקן, מעשר לבערך 8.16. היא לא גדלה.",
      do=show("q-x-2", "q-n-2"),
      say="ואחת מהן גם טועה בכיוון: ערך באמצע מקטין את סטיית התקן, מעשר לבערך שמונה נקודה שש עשרה. היא לא גדלה.",
      hold=0.6),
], stage=w.col(w.eq("x-1", "$10,\\ 20,\\ 30 \\;\\to\\; {-1},\\ 0,\\ 1$", 40),
               w.eq("x-2", "$10,\\ 20,\\ 30,\\ \\mathbf{20} \\quad \\min = 10,\\ \\max = 30$", 36),
               w.eq("x-3", "$\\to\\; {-1},\\ 0,\\ 1,\\ \\mathbf{0}$", 40),
               st.card("x-t", "trap", "⚠️ <span class='ev-l'>min-max</span> רואה רק את הקצוות, לא את סטיית התקן"),
               gap=26),
   panel=w.stem(Q["id"], "שלושה ערכים במרווחים שווים, <span class='ev-l'>minmax</span> ל-<span class='ev-l'>[−1, 1]</span>. נוסף ערך השווה לממוצע "
                         "המקורי, ומסלמים שוב. מה קורה?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={0: "לא תלוי ב-σ", 1: "לא תלוי ב-σ", 2: "σ קטנה"})
         + "</div>")
exam.phrases[4].do += show("x-t")

recap = Scene("recap", "סיכום", [
    P("לסיכום: מרחקים וסכומים משוקללים צריכים סילום. סף על מאפיין אחד, כמו בעץ, לא.", do=show("r-1")),
    P("min-max רואה רק את המינימום ואת המקסימום. z-score מחסיר את הממוצע ומחלק בסטיית התקן.", do=show("r-2"),
      say="מִין מַקְס רואה רק את המינימום ואת המקסימום. זֶד סְקוֹר מחסיר את הממוצע ומחלק בסטיית התקן."),
    P("סטיית התקן בקורס: חלוקה ב-n פחות אחת.", do=show("r-3"), say="סטיית התקן בקורס: חלוקה באֶן פחות אחת."),
    P("וחריג: מודדים מרחק בסטיות תקן ומשווים לסף. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(w.eq("r-e", "$\\sigma = \\sqrt{\\dfrac{\\sum (x - \\bar{x})^2}{n - 1}}$", 48, hidden=False)),
   panel=st.card("r-1", "rule", "🔑 מרחק / סכום משוקלל ← סילום")
         + st.card("r-2", "rule", "🔑 $\\dfrac{x - \\min}{\\max - \\min}$ · $\\dfrac{x - \\bar{x}}{\\sigma}$")
         + st.card("r-3", "rule", "🔑 חלוקה ב-$n - 1$")
         + st.card("r-4", "trap", "⚠️ $\\dfrac{|x - \\bar{x}|}{\\sigma}$ מול הסף: קטן ← משאירים"))

VIDEO = w.check_say(Video("E08", "eval", "סילום, סטיית תקן וחריגים", [title, why, two, out, exam, recap],
                          bank={"SAMP-1-Q2": 3}))
