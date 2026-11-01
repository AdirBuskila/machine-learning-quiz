# -*- coding: utf-8 -*-
"""E05 · Multi-class: macro ו-micro — video script (build: tools/video/vid.py E05).

Source: courses/eval/E05.js (goal: precision and recall per class, macro averages, micro = accuracy, and rejecting
"right number, wrong name" options). Exam anchor: 24B-C-Q6 (action / drama series, 115 action with 95 caught and 25
drama called action), which no guided quiz uses; its numbers appear in no quiz item.
Not used on purpose: the E05 quiz scenarios (dog / cat 60/40 with 54/8, accuracy 82% -> micro, Barcelona / Real
80/70 with 72/24, the 0.9 / 0.5 weighted average) and the refs 24S-A-Q1, 25B-A-Q14; the boss refs 24S-C-Q5 and the
E10 Liverpool question; the lesson's apples. 26B-A-Q1 and 25S-A-Q5 are kept for the E10 walkthrough.
Worked example: crows / pigeons, 50 crows (44 called crow, 6 pigeon) and 20 pigeons (1 called crow, 19 pigeon):
accuracy 90%, macro-P ≈ 86.9%, macro-R = 91.5%. Exact fractions, cross-checked with sklearn (macro and micro too).
"""
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

Fr = w.Fr
# ---------------------------------------------------------------- crows / pigeons (verified)
M = w.metrics(44, 6, 1, 19)
w.sk_check(M)
assert M["n"] == 70 and M["acc"] == Fr(9, 10) and M["micro"] == M["acc"]
assert M["prec"] == Fr(44, 45) and M["rec"] == Fr(22, 25)               # crow: ≈97.8%, 88%
assert M["prec_n"] == Fr(19, 25) and M["rec_n"] == Fr(19, 20)           # pigeon: 76%, 95%
assert w.fr_pct(M["prec"], 1) == "97.8" and w.fr_pct(M["macro_p"], 1) == "86.9" and M["macro_r"] == Fr(183, 200)
assert M["macro_p"] < M["acc"] < M["macro_r"]

# ---------------------------------------------------------------- exam 24B-C-Q6 (verified)
Q = BANK["24B-C-Q6"]
assert Q["correctIndex"] == 2 and "115" in Q["question"] and "95" in Q["question"] and "25" in Q["question"]
X = w.metrics(95, 115 - 95, 25, (200 - 115) - 25)
w.sk_check(X)
assert (X["fn"], X["tn"], X["n"]) == (20, 60, 200)
assert w.fr_pct(X["err"], 1) == "22.5" and w.fr_pct(X["acc"], 1) == "77.5"           # options say 20.5 and 75.5
assert w.fr_pct(X["rec"], 1) == "82.6" and w.fr_pct(X["rec_n"], 1) == "70.6"
assert w.fr_pct(X["macro_r"], 1) == "76.6"                                           # the key
assert w.fr_pct(X["prec"], 1) == "79.2" and X["prec_n"] == Fr(3, 4) and w.fr_pct(X["macro_p"], 1) == "77.1"  # not 77.7
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
CM = w.cmatrix("cm", 44, 6, 1, 19, pos="עורב", neg="יונה", size=30, tags=False)
CX = w.cmatrix("cx", 95, 20, 25, 60, pos="פעולה", neg="דרמה", size=24, tags=False, hidden=True)
L = lambda t: f"<span class='ev-l'>{t}</span>"
ROWS = [["עורב", L("44/45 ≈ 97.8%"), L("44/50 = 88%")], ["יונה", L("19/25 = 76%"), L("19/20 = 95%")],
        [L("<b>macro</b>"), L("<b>≈ 86.9%</b>"), L("<b>91.5%</b>")]]
HEAD = ["מחלקה", L("Precision"), L("Recall")]
TAB = w.table("pc", HEAD, ROWS[:2], size=28)
TAB3 = w.table("pd", HEAD, ROWS, size=28, hide_rows=False)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נמדוד מודל עם כמה מחלקות שחשובות באותה מידה: Precision ו-Recall לכל מחלקה, ואז ממוצע.",
      say="בסרטון הזה נמדוד מודל עם כמה מחלקות שחשובות באותה מידה: פְּרֶסִיזְ'ן ורִיקוֹל לכל מחלקה, ואז ממוצע."),
    P("נכיר את macro ואת micro, ונפתור שאלת מבחן שכל התשובות בה נראות סבירות.", hold=0.4,
      say="נכיר את המַקְרוֹ ואת המַיְקְרוֹ, ונפתור שאלת מבחן שכל התשובות בה נראות סבירות."),
], stage='<div class="kicker">E05 · הערכת מודל</div><h1>Multi-class: macro ו-micro</h1>'
         '<p>מדדים לכל מחלקה, ממוצע macro, ולמה micro שווה ל-accuracy.</p>', layout="title")

per = Scene("per", "מדד לכל מחלקה", [
    P("מודל מזהה ציפורים בתמונות: עורב או יונה, ושתי הטעויות חמורות באותה מידה.", do=show("cm")),
    P("לכן אין כאן מחלקה ״חיובית״ אחת. מחשבים את המדדים **לכל מחלקה בנפרד**."),
    P("עורב: המודל אמר עורב 45 פעמים, וצדק ב-44. זה ה-Precision של עורב.", do=show("cm-h-tp", "cm-h-fp", "pc", "pc-r0"),
      say="עורב: המודל אמר עורב ארבעים וחמש פעמים, וצדק בארבעים וארבע. זה הפְּרֶסִיזְ'ן של עורב."),
    P("ויש 50 עורבים, ו-44 זוהו: Recall של 88 אחוז.", do=hide("cm-h-fp") + show("cm-h-fn"),
      say="ויש חמישים עורבים, וארבעים וארבעה זוהו: רִיקוֹל של שמונים ושמונה אחוז."),
    P("עכשיו יונה. התפקידים מתהפכים: התא הימני התחתון הוא הפגיעות שלה.", do=hide("cm-h-tp", "cm-h-fn") + show("cm-h-tn")),
    P("המודל אמר יונה 25 פעמים: 19 יונים, ו-6 עורבים. 19 חלקי 25, 76 אחוז.", do=show("cm-h-fn", "pc-r1"),
      say="המודל אמר יונה עשרים וחמש פעמים: תשע עשרה יונים, ושישה עורבים. תשע עשרה חלקי עשרים וחמש, שבעים ושישה אחוז."),
    P("והפספוסים של העורב הם אזעקות השווא של היונה. כל טעות נספרת פעם לכל מחלקה.", do=show("p-k"), hold=0.4),
], stage=w.col(CM, TAB, gap=26),
   panel=st.card("p-1", "plain", "עורב: הפגיעות = התא השמאלי העליון")
         + st.card("p-2", "plain", "יונה: הפגיעות = התא הימני התחתון")
         + st.card("p-k", "rule", "🔑 מה שהוא <span class='ev-l'>FN</span> לעורב הוא <span class='ev-l'>FP</span> ליונה"))
per.phrases[2].do += show("p-1")
per.phrases[4].do += show("p-2")
# the matrix is RTL: column "חזוי עורב" is on the right; say it by content, the cards name the cells by their role
per.phrases[4].sub = "עכשיו יונה. התפקידים מתהפכים: התא של ״באמת יונה, חזוי יונה״ הוא הפגיעות שלה."
per.panel = per.panel.replace("התא השמאלי העליון", "באמת עורב · חזוי עורב").replace("התא הימני התחתון",
                                                                                     "באמת יונה · חזוי יונה")

macro = Scene("macro", "macro ו-micro", [
    P("macro הוא ממוצע **פשוט** של המחלקות. Precision: 97.8 ו-76, הממוצע בערך 86.9 אחוז.",
      do=show("m-1"),
      say="מַקְרוֹ הוא ממוצע פשוט של המחלקות. פְּרֶסִיזְ'ן: תשעים ושבע נקודה שמונה ושבעים ושש, הממוצע בערך שמונים ושש נקודה תשע אחוז."),
    P("Recall: 88 ו-95, הממוצע 91.5 אחוז.", do=show("m-2"),
      say="רִיקוֹל: שמונים ושמונה ותשעים וחמש, הממוצע תשעים ואחת נקודה חמש אחוז."),
    P("שימו לב: יש רק 20 יונים מול 50 עורבים, ובכל זאת היונה שוקלת חצי. זו כל הנקודה של macro.", do=show("m-k"),
      say="שימו לב: יש רק עשרים יונים מול חמישים עורבים, ובכל זאת היונה שוקלת חצי. זו כל הנקודה של מַקְרוֹ."),
    P("micro עושה הפוך: מאחד את כל ההחלטות לספירה אחת, ומחשב פעם אחת.", do=show("m-3"),
      say="מַיְקְרוֹ עושה הפוך: מאחד את כל ההחלטות לספירה אחת, ומחשב פעם אחת."),
    P("הפגיעות של כולם: 44 ועוד 19, 63. וכל טעות היא אזעקת שווא של מחלקה אחת, ופספוס של השנייה.", do=show("cm-h-tp", "cm-h-tn"),
      say="הפגיעות של כולם: ארבעים וארבע ועוד תשע עשרה, שישים ושלוש. וכל טעות היא אזעקת שווא של מחלקה אחת, ופספוס של השנייה."),
    P("לכן micro-Precision ו-micro-Recall יוצאים 63 חלקי 70, 90 אחוז. בדיוק ה-accuracy.", do=show("m-4"), hold=0.4,
      say="לכן המַיְקְרוֹ פְּרֶסִיזְ'ן והמַיְקְרוֹ רִיקוֹל יוצאים שישים ושלוש חלקי שבעים, תשעים אחוז. בדיוק האַקְיוּרַסִי."),
], stage=w.col(CM.replace('id="cm', 'id="cn'), TAB3, gap=26),
   panel=st.card("m-1", "ex", "$\\text{macro-}P = \\dfrac{97.8 + 76}{2} \\approx 86.9\\%$")
         + st.card("m-2", "ex", "$\\text{macro-}R = \\dfrac{88 + 95}{2} = 91.5\\%$")
         + st.card("m-k", "rule", "🔑 <span class='ev-l'>macro</span>: כל מחלקה שוקלת אותו דבר")
         + st.card("m-3", "plain", "<span class='ev-l'>micro</span>: כל ההחלטות יחד")
         + st.card("m-4", "rule", "🔑 <span class='ev-l'>micro-P = micro-R = Accuracy</span><br>"
                                  "$\\dfrac{44 + 19}{70} = 90\\%$"))
for p in macro.phrases:   # the matrix / table of this scene are copies with their own ids
    p.do = [(k, i.replace("pc", "pd").replace("cm-", "cn-")) for k, i in p.do]

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: מודל מסווג סדרות לפעולה או דרמה, multi-class. 200 סדרות, 115 מהן פעולה, ו-95 זוהו.",
      do=show("cx"),
      say="שאלה ממבחן: מודל מסווג סדרות לפעולה או דרמה, כבעיית מוּלְטִי קְלַאס. מאתיים סדרות, מאה וחמש עשרה מהן פעולה, ותשעים וחמש זוהו."),
    P("25 דרמות סווגו כפעולה. אז 20 סדרות פעולה חמקו, ומתוך 85 הדרמות 60 סווגו נכון.",
      say="עשרים וחמש דרמות סווגו כפעולה. אז עשרים סדרות פעולה חמקו, ומתוך שמונים וחמש הדרמות, שישים סווגו נכון."),
    P("לפני שמסתכלים על התשובות, מחשבים הכול. הדיוק: 155 מתוך 200, 77.5 אחוז. השגיאה: 22.5 אחוז.", do=show("x-1"),
      say="לפני שמסתכלים על התשובות, מחשבים הכול. הדיוק: מאה חמישים וחמש מתוך מאתיים, שבעים ושבע נקודה חמש אחוז. השגיאה: עשרים ושתיים נקודה חמש אחוז."),
    P("Recall: פעולה 95 מתוך 115, ודרמה 60 מתוך 85. הממוצע: בערך 76.6 אחוז.", do=show("x-2"),
      say="רִיקוֹל: פעולה תשעים וחמש מתוך מאה וחמש עשרה, ודרמה שישים מתוך שמונים וחמש. הממוצע: בערך שבעים ושש נקודה שש אחוז."),
    P("Precision: פעולה 95 מתוך 120, ודרמה 60 מתוך 80. הממוצע: בערך 77.1 אחוז.", do=show("x-3"),
      say="פְּרֶסִיזְ'ן: פעולה תשעים וחמש מתוך מאה ועשרים, ודרמה שישים מתוך שמונים. הממוצע: בערך שבעים ושבע נקודה אחת אחוז."),
    P("עכשיו התשובות. macro-Recall של 76.6 אחוז: מתאים בדיוק. זו התשובה.", do=show("q-ok-2"),
      say="עכשיו התשובות. מַקְרוֹ רִיקוֹל של שבעים ושש נקודה שש אחוז: מתאים בדיוק. זו התשובה."),
    P("השגיאה היא 22.5, לא 20.5. הדיוק הוא 77.5, לא 75.5. ו-macro-Precision יוצא 77.1, לא 77.7.",
      do=show("q-x-0", "q-n-0", "q-x-1", "q-n-1", "q-x-3", "q-n-3"),
      say="השגיאה היא עשרים ושתיים נקודה חמש, לא עשרים נקודה חמש. הדיוק הוא שבעים ושבע נקודה חמש, לא שבעים וחמש נקודה חמש. והמַקְרוֹ פְּרֶסִיזְ'ן יוצא שבעים ושבע נקודה אחת, לא שבעים ושבע נקודה שבע."),
    P("כאן המסיחים קרובים מאוד לאמת. רק מי שחישב את כל ארבעת המספרים עד הסוף, בטוח בתשובה.", do=show("x-t"), hold=0.6),
], stage=w.col(CX,
               w.eq("x-1", "$\\text{Acc} = \\dfrac{95 + 60}{200} = 77.5\\% \\qquad \\text{Err} = 22.5\\%$", 28),
               w.eq("x-2", "$\\text{macro-}R = \\tfrac{1}{2}\\left(\\tfrac{95}{115} + \\tfrac{60}{85}\\right) \\approx 76.6\\%$", 28),
               w.eq("x-3", "$\\text{macro-}P = \\tfrac{1}{2}\\left(\\tfrac{95}{120} + \\tfrac{60}{80}\\right) \\approx 77.1\\%$", 28),
               gap=14),
   panel=w.stem(Q["id"], "פעולה או דרמה, <span class='ev-l'>multi-class</span>. 200 סדרות: 115 פעולה (95 זוהו), "
                         "ו-25 דרמות סווגו כפעולה. איזו טענה נכונה?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={0: "22.5%", 1: "77.5%", 3: "77.1%"}) + "</div>"
         + st.card("x-t", "trap", "⚠️ מחשבים את <b>כל</b> המדדים, ורק אז משווים"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: כשהמחלקות חשובות באותה מידה, מחשבים Precision ו-Recall לכל מחלקה.", do=show("r-1"),
      say="לסיכום: כשהמחלקות חשובות באותה מידה, מחשבים פְּרֶסִיזְ'ן ורִיקוֹל לכל מחלקה."),
    P("במחלקה השנייה התפקידים מתהפכים: הפספוסים של האחת הם אזעקות השווא של השנייה.", do=show("r-2")),
    P("macro הוא ממוצע פשוט, וכל מחלקה שוקלת אותו דבר. micro שווה לדיוק.", do=show("r-3"),
      say="מַקְרוֹ הוא ממוצע פשוט, וכל מחלקה שוקלת אותו דבר. מַיְקְרוֹ שווה לדיוק."),
    P("ובמבחן: מחשבים את כל המספרים לפני שקוראים את התשובות. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(w.cmatrix("cr", "", "", "", "", pos="A", neg="B", size=26)),
   panel=st.card("r-1", "rule", "🔑 $P$ ו-$R$ <b>לכל מחלקה</b>")
         + st.card("r-2", "rule", "🔑 $P_B = \\dfrac{TN}{TN + FN}$ · $R_B = \\dfrac{TN}{TN + FP}$")
         + st.card("r-3", "rule", "🔑 <span class='ev-l'>macro</span> = ממוצע פשוט · <span class='ev-l'>micro</span> = "
                                  "<span class='ev-l'>Accuracy</span>")
         + st.card("r-4", "trap", "⚠️ מספר נכון עם שם שגוי, או מספר כמעט נכון"))

VIDEO = w.check_say(Video("E05", "eval", "Multi-class: macro ו-micro", [title, per, macro, exam, recap],
                          bank={"24B-C-Q6": 2}))
