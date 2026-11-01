# -*- coding: utf-8 -*-
"""E04 · Precision, Recall ו-F1 — video script (build: tools/video/vid.py E04).

Source: courses/eval/E04.js (goal: compute precision, recall and F1 from a matrix or a story, and read a gap between
them in terms of FP and FN). Exam part: an original "בסגנון מבחן" question (bank={}): every free bank question on
this skill is a twin of a quiz item (23B-B-Q10 / 23S-A-Q1 / 24B-A-Q9 share the ratios of 23B-A-Q10; 26S-A-Q11 and
CLAUDE-26-Q5 repeat the gap questions 26B-A-Q6 / 25B-A-Q7; CLAUDE-26-Q19 repeats the F1 quiz item), and PRAC-X-Q17
is solved in E03.
Not used on purpose: the E04 quiz scenarios (the disease test 40/160 with 32/16, the spam filter 300/700 with
240/40, P = 0.6 / R = 0.9, the matrix 70/30/10/90, the gaps 0.95/0.4 and 0.5/0.98, 50/10/10 plus 20 FP) and the
refs 23B-A-Q10, SAMP-1-Q3, 25B-A-Q7; the lesson's apples and the 90/10/30/50 family.
Worked examples: a blurry-photo detector (TP 36, FN 12, FP 4, TN 148) and two settings of it (30/18/2 and 46/2/30);
the exam-style question: 360 chips, 90 defective, 45 caught, 5 false alarms. All exact, cross-checked with sklearn.
"""
from vid import P, Scene, Video, show, hide
import stage as st
import w_eval as w

Fr = w.Fr
# ---------------------------------------------------------------- blurry photos (verified)
M = w.metrics(36, 12, 4, 148)
w.sk_check(M)
assert M["n"] == 200 and M["prec"] == Fr(9, 10) and M["rec"] == Fr(3, 4)
assert M["f1"] == Fr(9, 11) and w.fr_dec(M["f1"]) == "0.82"
assert (M["prec"] + M["rec"]) / 2 == Fr(33, 40) and M["f1"] < Fr(33, 40)        # harmonic < arithmetic (0.825)
assert M["acc"] == Fr(23, 25)

# two settings of the same detector, same 48 blurry photos
A = w.metrics(30, 18, 2, 150)
B = w.metrics(46, 2, 30, 122)
for X in (A, B):
    w.sk_check(X)
    assert X["tp"] + X["fn"] == 48 and X["n"] == 200
assert w.fr_pct(A["prec"], 1) == "93.8" and w.fr_pct(A["rec"], 1) == "62.5"
assert w.fr_pct(B["prec"], 1) == "60.5" and w.fr_pct(B["rec"], 1) == "95.8"
assert A["prec"] > A["rec"] and B["rec"] > B["prec"]

# ---------------------------------------------------------------- the exam-style question (verified)
C = w.metrics(45, 45, 5, 265)
w.sk_check(C)
assert C["n"] == 360 and C["tp"] + C["fn"] == 90 and C["fp"] + C["tn"] == 270
assert C["prec"] == Fr(9, 10) and C["rec"] == Fr(1, 2)
assert w.fr_pct(C["f1"], 1) == "64.3" and C["f1"] == Fr(9, 14)
assert (C["prec"] + C["rec"]) / 2 == Fr(7, 10)                                   # the "70%" distractor
assert w.fr_pct(C["acc"], 1) == "86.1"
XO = [("a", "ה-<span class='ev-l'>F1</span> הוא כ-64.3%"), ("b", "ה-<span class='ev-l'>F1</span> הוא 70%"),
      ("c", "ה-<span class='ev-l'>recall</span> הוא 90%"), ("d", "ה-<span class='ev-l'>accuracy</span> הוא 50%")]

# ---------------------------------------------------------------- figures
CM = w.cmatrix("cm", 36, 12, 4, 148, pos="מטושטשת", neg="חדה", size=30)
CA = w.cmatrix("ka", 30, 18, 2, 150, pos="מטושטשת", neg="חדה", size=22, tags=False)
CB = w.cmatrix("kb", 46, 2, 30, 122, pos="מטושטשת", neg="חדה", size=22, tags=False)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד שני מדדים שמסתכלים רק על המחלקה החיובית, כל אחד מכיוון אחר: Precision ו-Recall.",
      say="בסרטון הזה נלמד שני מדדים שמסתכלים רק על המחלקה החיובית, כל אחד מכיוון אחר: פְּרֶסִיזְ'ן ורִיקוֹל."),
    P("נאחד אותם ל-F1, ונבין מה אומר פער גדול ביניהם.", hold=0.4, say="נאחד אותם לאֶף וַואן, ונבין מה אומר פער גדול ביניהם."),
], stage='<div class="kicker">E04 · הערכת מודל</div><h1>Precision, Recall ו-F1</h1>'
         '<p>לחשב את שלושתם ממטריצה או מסיפור, ולפרש פער ביניהם במונחי FP ו-FN.</p>', layout="title")

two = Scene("two", "שתי שאלות", [
    P("מודל מסמן תמונות מטושטשות באלבום. מתוך מאתיים תמונות, ארבעים ושמונה באמת מטושטשות. החיובי: מטושטשת.",
      do=show("cm")),
    P("**Precision** שואל: מכל מה שהמודל **סימן**, כמה באמת מטושטשות? מסתכלים על העמודה של ״חזוי מטושטשת״.",
      do=show("cm-h-tp", "cm-h-fp"),
      say="פְּרֶסִיזְ'ן שואל: מכל מה שהמודל סימן, כמה באמת מטושטשות? מסתכלים על העמודה של חזוי מטושטשת."),
    P("הוא סימן 40 תמונות, ו-36 מהן באמת מטושטשות. 36 חלקי 40: 0.9.", do=show("e-p"),
      say="הוא סימן ארבעים תמונות, ושלושים ושש מהן באמת מטושטשות. שלושים ושש חלקי ארבעים: אפס נקודה תשע."),
    P("**Recall** שואל: מכל המטושטשות **שקיימות**, כמה המודל תפס? עכשיו מסתכלים על השורה.",
      do=hide("cm-h-fp") + show("cm-h-fn"),
      say="רִיקוֹל שואל: מכל המטושטשות שקיימות, כמה המודל תפס? עכשיו מסתכלים על השורה."),
    P("יש 48, והוא תפס 36. 36 חלקי 48: 0.75.", do=show("e-r"),
      say="יש ארבעים ושמונה, והוא תפס שלושים ושש. שלושים ושש חלקי ארבעים ושמונה: אפס נקודה שבעים וחמש."),
    P("המונה זהה. ההבדל הוא רק מה מוסיפים במכנה: אזעקות שווא, או פספוסים.", do=hide("cm-h-tp", "cm-h-fn") + show("t-k"),
      hold=0.3),
    P("לכן כל אזעקת שווא מורידה רק את Precision, וכל פספוס מוריד רק את Recall.", do=show("t-k2"),
      say="לכן כל אזעקת שווא מורידה רק את הפְּרֶסִיזְ'ן, וכל פספוס מוריד רק את הרִיקוֹל.", hold=0.4),
], stage=w.col(CM, w.eq("e-p", "$\\text{Precision} = \\dfrac{TP}{TP + FP} = \\dfrac{36}{36 + 4} = 0.9$", 32),
               w.eq("e-r", "$\\text{Recall} = \\dfrac{TP}{TP + FN} = \\dfrac{36}{36 + 12} = 0.75$", 32), gap=16),
   panel=st.card("t-k", "rule", "🔑 <span class='ev-l'>Precision</span>: חלקי <b>מה שסומן</b> (עמודה)<br>"
                                 "🔑 <span class='ev-l'>Recall</span>: חלקי <b>מה שקיים</b> (שורה)")
         + st.card("t-k2", "rule", "🔑 <span class='ev-l'>FP</span> פוגע ב-<span class='ev-l'>Precision</span> · "
                                   "<span class='ev-l'>FN</span> פוגע ב-<span class='ev-l'>Recall</span>"))

f1 = Scene("f1", "F1", [
    P("רוצים מספר אחד שמשלב את שניהם? זה F1: פעמיים המכפלה, חלקי הסכום.", do=show("f-1"),
      say="רוצים מספר אחד שמשלב את שניהם? זה אֶף וַואן: פעמיים המכפלה, חלקי הסכום."),
    P("פעמיים 0.9 כפול 0.75 זה 1.35. הסכום הוא 1.65. והמנה: בערך 0.82.", do=show("f-2"),
      say="פעמיים אפס נקודה תשע כפול אפס נקודה שבעים וחמש זה אחת נקודה שלושים וחמש. הסכום הוא אחת נקודה שישים וחמש. והמנה: בערך אפס נקודה שמונים ושתיים."),
    P("הממוצע הרגיל היה יוצא 0.825. F1 קצת נמוך ממנו, כי זה ממוצע **הרמוני**: הוא נמשך אל המדד הנמוך.",
      do=show("f-3"),
      say="הממוצע הרגיל היה יוצא אפס נקודה שמונה מאות עשרים וחמש. אֶף וַואן קצת נמוך ממנו, כי זה ממוצע הרמוני: הוא נמשך אל המדד הנמוך."),
    P("וככל ששני המדדים רחוקים יותר זה מזה, הפער בין שני הממוצעים גדל. עוד רגע נראה את זה.", hold=0.4),
], stage=w.col(w.eq("f-1", "$F_1 = \\dfrac{2 \\cdot P \\cdot R}{P + R}$", 46),
               w.eq("f-2", "$F_1 = \\dfrac{2 \\cdot 0.9 \\cdot 0.75}{0.9 + 0.75} = \\dfrac{1.35}{1.65} \\approx 0.82$", 38),
               w.eq("f-3", "$\\dfrac{0.9 + 0.75}{2} = 0.825$", 34), gap=30),
   panel=st.card("f-k", "rule", "🔑 <span class='ev-l'>F1</span> = ממוצע <b>הרמוני</b> של <span class='ev-l'>P</span> ו-"
                                 "<span class='ev-l'>R</span>, תמיד בין שניהם ולא מעל הממוצע הרגיל"))
f1.phrases[2].do += show("f-k")

gap = Scene("gap", "פער בין המדדים", [
    P("עכשיו נכוון את אותו מודל פעמיים. בגרסה הזהירה, הוא מסמן תמונה רק כשהוא בטוח מאוד.", do=show("ka", "g-a")),
    P("רק שתי אזעקות שווא, ולכן Precision גבוה: כמעט 94 אחוז. אבל 18 מטושטשות חמקו, ו-Recall יורד ל-62.5 אחוז.",
      do=show("ka-h-fp", "ka-h-fn", "g-a2"),
      say="רק שתי אזעקות שווא, ולכן הפְּרֶסִיזְ'ן גבוה: כמעט תשעים וארבעה אחוז. אבל שמונה עשרה מטושטשות חמקו, והרִיקוֹל יורד לשישים ושתיים נקודה חמש אחוז."),
    P("בגרסה הנדיבה, הוא מסמן כל תמונה חשודה.", do=hide("ka-h-fp", "ka-h-fn") + show("kb", "g-b")),
    P("רק שתי מטושטשות חמקו, אז Recall כמעט 96 אחוז. אבל שלושים אזעקות שווא מורידות את Precision ל-60.5 אחוז.",
      do=show("kb-h-fn", "kb-h-fp", "g-b2"),
      say="רק שתי מטושטשות חמקו, אז הרִיקוֹל כמעט תשעים ושישה אחוז. אבל שלושים אזעקות שווא מורידות את הפְּרֶסִיזְ'ן לשישים נקודה חמש אחוז."),
    P("איך קוראים פער? המדד הנמוך מצביע על התא האשם: Precision נמוך, תסתכלו על FP. Recall נמוך, על FN.",
      do=hide("kb-h-fn", "kb-h-fp") + show("g-k"),
      say="איך קוראים פער? המדד הנמוך מצביע על התא האשם: פְּרֶסִיזְ'ן נמוך, תסתכלו על אזעקות השווא. רִיקוֹל נמוך, על הפספוסים."),
    P("ואיזו גרסה עדיפה? תלוי מה יקר יותר בבעיה: אזעקת שווא, או פספוס.", hold=0.4),
], stage=w.row(w.col(CB, w.chip("g-b", "נדיב", "warn"), gap=8), w.col(CA, w.chip("g-a", "זהיר", "info"), gap=8), gap=30),
   panel=st.card("g-a2", "ex", "זהיר: $P \\approx 93.8\\%$ · $R = 62.5\\%$")
         + st.card("g-b2", "ex", "נדיב: $P \\approx 60.5\\%$ · $R \\approx 95.8\\%$")
         + st.card("g-k", "rule", "🔑 <span class='ev-l'>Precision</span> נמוך ← הרבה <span class='ev-l'>FP</span><br>"
                                  "🔑 <span class='ev-l'>Recall</span> נמוך ← הרבה <span class='ev-l'>FN</span>"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: 360 שבבים, 90 מהם פגומים. המערכת תפסה 45 מהפגומים, וסימנה בטעות 5 שבבים תקינים.",
      say="שאלה בסגנון מבחן: שלוש מאות שישים שבבים, תשעים מהם פגומים. המערכת תפסה ארבעים וחמישה מהפגומים, וסימנה בטעות חמישה שבבים תקינים."),
    P("קודם המטריצה: 45 נתפסו ו-45 חמקו. 5 אזעקות שווא, ומהתקינים נשארים 265.", do=show("cx"),
      say="קודם המטריצה: ארבעים וחמישה נתפסו, וארבעים וחמישה חמקו. חמש אזעקות שווא, ומהתקינים נשארים מאתיים שישים וחמישה."),
    P("Precision: 45 מתוך 50 שסומנו, 0.9. Recall: 45 מתוך 90 קיימים, 0.5.", do=show("x-1"),
      say="פְּרֶסִיזְ'ן: ארבעים וחמישה מתוך חמישים שסומנו, אפס נקודה תשע. רִיקוֹל: ארבעים וחמישה מתוך תשעים קיימים, אפס נקודה חמש."),
    P("F1: פעמיים 0.45, חלקי 1.4. יוצא בערך 64.3 אחוז. זו התשובה.", do=show("x-2", "q-ok-a"),
      say="אֶף וַואן: פעמיים אפס נקודה ארבעים וחמש, חלקי אחת נקודה ארבע. יוצא בערך שישים וארבע נקודה שלוש אחוז. זו התשובה."),
    P("70 אחוז הוא הממוצע הרגיל של 0.9 ו-0.5. עם פער כזה, הממוצע ההרמוני נמוך ממנו בהרבה.",
      do=show("q-x-b", "q-n-b"),
      say="שבעים אחוז הוא הממוצע הרגיל של אפס נקודה תשע ואפס נקודה חמש. עם פער כזה, הממוצע ההרמוני נמוך ממנו בהרבה."),
    P("90 אחוז הוא Precision, לא Recall. ו-50 אחוז הוא Recall, לא דיוק: הדיוק הוא 310 מתוך 360, בערך 86 אחוז.",
      do=show("q-x-c", "q-n-c", "q-x-d", "q-n-d"),
      say="תשעים אחוז הוא הפְּרֶסִיזְ'ן, לא הרִיקוֹל. וחמישים אחוז הוא הרִיקוֹל, לא הדיוק: הדיוק הוא שלוש מאות ועשר מתוך שלוש מאות שישים, בערך שמונים ושישה אחוז."),
    P("שוב אותה מלכודת: מספרים אמיתיים, עם השם הלא נכון. מחשבים הכול, ורק אז מתאימים.", do=show("x-t"), hold=0.6),
], stage=w.col(w.cmatrix("cx", 45, 45, 5, 265, pos="פגום", neg="תקין", size=26, hidden=True),
               w.eq("x-1", "$P = \\dfrac{45}{50} = 0.9 \\qquad R = \\dfrac{45}{90} = 0.5$", 32),
               w.eq("x-2", "$F_1 = \\dfrac{2 \\cdot 0.9 \\cdot 0.5}{0.9 + 0.5} = \\dfrac{0.9}{1.4} \\approx 0.643$", 32), gap=16),
   panel=w.stem("בסגנון מבחן", "360 שבבים, 90 פגומים. נתפסו 45 מהפגומים, ו-5 תקינים סומנו בטעות (החיובי = פגום). "
                               "איזו טענה נכונה?")
         + '<div class="ev-opts">' + st.options("q", XO, notes={"b": "ממוצע רגיל", "c": "זה P", "d": "זה R"}) + "</div>"
         + st.card("x-t", "trap", "⚠️ מספר נכון, שם שגוי"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: Precision הוא החיוביים האמיתיים מתוך כל מה שסומן. העמודה.", do=show("r-1"),
      say="לסיכום: פְּרֶסִיזְ'ן הוא החיוביים האמיתיים מתוך כל מה שסומן. העמודה."),
    P("Recall הוא החיוביים האמיתיים מתוך כל מה שקיים. השורה.", do=show("r-2"),
      say="רִיקוֹל הוא החיוביים האמיתיים מתוך כל מה שקיים. השורה."),
    P("F1 הוא הממוצע ההרמוני שלהם, ונמשך אל הנמוך.", do=show("r-3"), say="אֶף וַואן הוא הממוצע ההרמוני שלהם, ונמשך אל הנמוך."),
    P("ובפער: המדד הנמוך מצביע על אזעקות שווא או על פספוסים. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(w.cmatrix("cr", "TP", "FN", "FP", "TN", tags=False)),
   panel=st.card("r-1", "rule", "🔑 $P = \\dfrac{TP}{TP + FP}$")
         + st.card("r-2", "rule", "🔑 $R = \\dfrac{TP}{TP + FN}$")
         + st.card("r-3", "rule", "🔑 $F_1 = \\dfrac{2PR}{P + R}$")
         + st.card("r-4", "trap", "⚠️ <span class='ev-l'>P</span> נמוך ← <span class='ev-l'>FP</span> · "
                                  "<span class='ev-l'>R</span> נמוך ← <span class='ev-l'>FN</span>"))

VIDEO = w.check_say(Video("E04", "eval", "Precision, Recall ו-F1", [title, two, f1, gap, exam, recap], bank={}))
