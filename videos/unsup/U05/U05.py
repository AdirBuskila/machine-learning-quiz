# -*- coding: utf-8 -*-
"""U05 · חישוב TF-IDF — video script (build: tools/video/vid.py U05).

Source: courses/unsup/U05.js (lesson + goal). Exam part: an ORIGINAL "בסגנון מבחן" question (bank={}): working back
from a given TF-IDF to the number of occurrences (N = 256, df = 32, 40 words, TF-IDF 0.3 -> 4 times). Every real-exam
text question twins a U05 quiz item (26B-A-Q11 / 26S-A-Q16 ~ U05-Q8, 23S-A-Q17 ~ U05-Q6), and CLAUDE-26-Q15's key
gives U05-Q3's reasoning, so no bank question is shown.
Avoided on purpose: the quiz items of U05 (PRAC-N-Q11, PRAC-N-Q12 and the originals: N=32/df=4, 16 docs with 4 in 200,
50/50 with 5 in 20, the "+1" case 16/2 with 3 in 60, log-TF 1 vs 15 as a ratio, "when is TF-IDF high", a word in all
documents vs a rare word, "why log on TF"), PRAC-N-Q9 (its key is wrong, see the report) and the boss items PRAC-X-Q2
and U08-Q5. The video never states "frequent here, rare elsewhere", "no discriminating power" or "repetition is not
linear importance"; it only computes.
New data: 32 documents, one of 25 words, three words — "קונבולוציה" (2 times, df 2), "שכבה" (3, df 4), "רשת" (5, df 8)
giving 0.32, 0.36, 0.4; the same with the "+1" IDF (0.4, 0.48, 0.6); log-TF for 0, 1, 3, 7 occurrences (0, 1, 2, 3).
"""
from fractions import Fraction as F
import math
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

N, LEN = 32, 25
WORDS = [("קונבולוציה", 2, 2), ("שכבה", 3, 4), ("רשת", 5, 8)]       # (word, count in the document, df)
TF = {w: F(c, LEN) for w, c, _ in WORDS}
IDF = {w: math.log2(N / df) for w, _, df in WORDS}
TFIDF = {w: float(TF[w]) * IDF[w] for w, _, _ in WORDS}
assert {w: num(float(t)) for w, t in TF.items()} == {"קונבולוציה": "0.08", "שכבה": "0.12", "רשת": "0.2"}
assert IDF == {"קונבולוציה": 4.0, "שכבה": 3.0, "רשת": 2.0}
assert {w: num(v) for w, v in TFIDF.items()} == {"קונבולוציה": "0.32", "שכבה": "0.36", "רשת": "0.4"}
assert sorted(TFIDF, key=TFIDF.get) == ["קונבולוציה", "שכבה", "רשת"]   # the rarest word is not the heaviest here
PLUS = {w: float(TF[w]) * (IDF[w] + 1) for w, _, _ in WORDS}
assert {w: num(v) for w, v in PLUS.items()} == {"קונבולוציה": "0.4", "שכבה": "0.48", "רשת": "0.6"}
LOGS = {c: math.log2(1 + c) for c in (0, 1, 3, 7)}
assert LOGS == {0: 0.0, 1: 1.0, 3: 2.0, 7: 3.0}

# ---------------------------------------------------------------- exam: original, בסגנון מבחן
XN, XDF, XLEN, XW = 256, 32, 40, 0.3
XIDF = math.log2(XN / XDF)
XCOUNT = XW / XIDF * XLEN
assert XIDF == 3 and round(XCOUNT, 9) == 4
assert round(XW * XLEN, 9) == 12                                  # forgot the IDF
assert round(XW / (XIDF + 1) * XLEN, 9) == 3                      # used the "+1" IDF the question did not give
assert round(XW / XIDF, 9) == 0.1                                 # stopped at TF
OPTS = [(0, "4"), (1, "12"), (2, "3"), (3, "0.1")]

# ---------------------------------------------------------------- widgets
ROWS = [[w, str(c), num(float(TF[w])), str(df), num(IDF[w]), num(TFIDF[w])] for w, c, df in WORDS]
TAB = W.table("tf", ["מילה", "הופעות", "TF", "df", "IDF", "TF-IDF"], ROWS, cls="rtl",
              hide={(r, c) for r in range(3) for c in (2, 4, 5)}, hl={(2, 5)})
TAB2 = W.table("tp", ["מילה", "TF", "IDF + 1", "TF-IDF"],
               [[w, num(float(TF[w])), num(IDF[w] + 1), num(PLUS[w])] for w, _, _ in WORDS], cls="rtl",
               hide={(r, c) for r in range(3) for c in (2, 3)})
LOGT = W.table("lt", ["הופעות", "0", "1", "3", "7"], [["TF"] + [num(v) for v in LOGS.values()]],
               hide={(0, c) for c in range(1, 5)})

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("ב-Bag of Words כל הופעה שווה 1. בסרטון הזה ניתן לכל מילה משקל אמיתי: TF-IDF.",
      say="בבֶּג אוֹף ווֹרְדְס כל הופעה שווה אחת. בסרטון הזה ניתן לכל מילה משקל אמיתי: טִי אֶף, אַיי דִּי אֶף."),
    P("נחשב אותו צעד אחר צעד, ונראה מתי הוא גבוה ומתי הוא בדיוק 0."),
    P("ובסוף שאלה בסגנון מבחן, שבה הולכים מהמשקל אחורה.", hold=0.4),
], stage='<div class="kicker">U05 · אשכול, טקסט ותמונה</div><h1>חישוב TF-IDF</h1>'
         '<p>לחשב TF-IDF לפי הנוסחה שהשאלה נותנת, ולדעת מתי הוא גבוה, מתי הוא 0 ולמה שמים לוג.</p>', layout="title")

idea = Scene("idea", "שני רכיבים", [
    P("TF-IDF הוא מכפלה של שני רכיבים. הראשון, TF: כמה המילה שכיחה במסמך הזה.", do=show("i-tf")),
    P("בגרסה היחסית: מספר ההופעות חלקי אורך המסמך.", do=show("i-tf2")),
    P("השני, IDF: כמה המילה נדירה בכל הקורפוס.", do=show("i-idf")),
    P("$N$ הוא מספר המסמכים, ו-df הוא בכמה מסמכים המילה מופיעה. לא כמה פעמים, אלא בכמה מסמכים.",
      say="אֶן הוא מספר המסמכים, ודִּי אֶף הוא בכמה מסמכים המילה מופיעה. לא כמה פעמים, אלא בכמה מסמכים.",
      do=show("i-idf2", "i-c1")),
    P("מילה נדירה: היחס גדול, והלוג חיובי. מילה שבכל המסמכים: היחס הוא 1, והלוג של 1 הוא 0.",
      say="מילה נדירה: היחס גדול, והלוג חיובי. מילה שבכל המסמכים: היחס הוא אחת, והלוג של אחת הוא אפס.",
      do=show("i-c2")),
    P("ואת שניהם כופלים. כדאי לזכור כמה לוגים בבסיס 2: של 2 זה 1, של 4 זה 2, של 8 זה 3, ושל 16 זה 4.",
      say="ואת שניהם כופלים. כדאי לזכור כמה לוגים בבסיס שתיים: של שתיים זה אחת, של ארבע זה שתיים, של שמונה זה שלוש, ושל שש עשרה זה ארבע.",
      do=show("i-x", "i-log"), hold=0.4),
], stage=W.stack(W.big("i-tf", "$TF = \\dfrac{\\text{count}}{\\text{length}}$", size=46),
                 W.big("i-idf", "$IDF = \\log_2 \\dfrac{N}{df}$", size=46),
                 W.big("i-x", "$TF\\text{-}IDF = TF \\times IDF$", size=46),
                 W.table("i-log", ["x", "1", "2", "4", "8", "16", "32"],
                         [["log₂ x", "0", "1", "2", "3", "4", "5"]], cls="sm").replace('class="u-tab sm"', 'class="u-tab sm h"'),
                 gap=30),
   panel=st.card("i-tf2", "ex", "TF יחסי: הופעות ⁄ מילים במסמך")
         + st.card("i-idf2", "ex", "N = מסמכים בקורפוס · df = מסמכים שהמילה בהם")
         + st.card("i-c1", "trap", "⚠️ df סופר <b>מסמכים</b>, לא הופעות")
         + st.card("i-c2", "rule", "🔑 בכל המסמכים: df = N, ולכן IDF = 0"))

calc = Scene("calc", "חישוב", [
    P("קורפוס של 32 מסמכים. מסמך אחד בו באורך 25 מילים, ונבדוק בו שלוש מילים.", do=show("tf")),
    P("קונבולוציה מופיעה במסמך פעמיים. TF הוא 2 חלקי 25, כלומר 0.08.",
      say="קונבולוציה מופיעה במסמך פעמיים. טִי אֶף הוא שתיים חלקי עשרים וחמש, כלומר אפס נקודה אפס שמונה.",
      do=show("tf-0-2")),
    P("והיא מופיעה רק ב-2 מסמכים בקורפוס. IDF הוא לוג של 32 חלקי 2, לוג של 16, כלומר 4.",
      say="והיא מופיעה רק בשני מסמכים בקורפוס. אַיי דִּי אֶף הוא לוג של שלושים ושתיים חלקי שתיים, לוג של שש עשרה, כלומר ארבע.",
      do=show("tf-0-4")),
    P("המכפלה: 0.08 כפול 4, כלומר 0.32.", say="המכפלה: אפס נקודה אפס שמונה כפול ארבע, כלומר אפס נקודה שלושים ושתיים.",
      do=show("tf-0-5")),
    P("שכבה: 3 הופעות, TF של 0.12. היא ב-4 מסמכים: לוג של 8, כלומר 3. המכפלה 0.36.",
      say="שכבה: שלוש הופעות, טִי אֶף של אפס נקודה שתים עשרה. היא בארבעה מסמכים: לוג של שמונה, כלומר שלוש. המכפלה אפס נקודה שלושים ושש.",
      do=show("tf-1-2", "tf-1-4", "tf-1-5")),
    P("רשת: 5 הופעות, TF של 0.2. היא ב-8 מסמכים: לוג של 4, כלומר 2. המכפלה 0.4.",
      say="רשת: חמש הופעות, טִי אֶף של אפס נקודה שתיים. היא בשמונה מסמכים: לוג של ארבע, כלומר שתיים. המכפלה אפס נקודה ארבע.",
      do=show("tf-2-2", "tf-2-4", "tf-2-5", "tf-hl-2-5")),
    P("שימו לב: המילה הנדירה ביותר לא קיבלה את המשקל הגבוה. שני הגורמים משחקים, ורק המכפלה קובעת.",
      do=show("c-c1"), hold=0.4),
], stage=TAB,
   panel=st.card("c-c1", "trap", "⚠️ לא מנחשים לפי גורם אחד: מחשבים את שניהם וכופלים"))

variants = Scene("var", "קראו את הנוסחה", [
    P("זהירות: לא כל שאלה משתמשת באותה נוסחה. לפעמים מוסיפים 1 ל-IDF.", do=show("v-f")),
    P("קונבולוציה: 0.08 כפול 5, כלומר 0.4. שכבה: 0.12 כפול 4, כלומר 0.48.",
      say="קונבולוציה: אפס נקודה אפס שמונה כפול חמש, כלומר אפס נקודה ארבע. שכבה: אפס נקודה שתים עשרה כפול ארבע, כלומר אפס נקודה ארבעים ושמונה.",
      do=show("tp-0-2", "tp-0-3", "tp-1-2", "tp-1-3")),
    P("ורשת: 0.2 כפול 3, כלומר 0.6. כל המספרים השתנו, אז קוראים את הנוסחה לפני שמחשבים.",
      say="ורשת: אפס נקודה שתיים כפול שלוש, כלומר אפס נקודה שש. כל המספרים השתנו, אז קוראים את הנוסחה לפני שמחשבים.",
      do=show("tp-2-2", "tp-2-3", "v-c1"), hold=0.4),
], stage=W.stack(W.big("v-f", "$IDF = \\log_2 \\dfrac{N}{df} + 1$", size=44), TAB2, gap=40),
   panel=st.card("v-c1", "trap", "⚠️ עם 1+ כל ה-IDF גדלים ב-1, וכל המשקלים משתנים"))

logtf = Scene("logtf", "TF לוגריתמי", [
    P("ויש גם TF לוגריתמי: לוג בבסיס 2 של אחת ועוד מספר ההופעות.",
      say="ויש גם טִי אֶף לוגריתמי: לוג בבסיס שתיים של אחת ועוד מספר ההופעות.", do=show("v-l", "lt")),
    P("אפס הופעות: לוג של 1, כלומר 0. הופעה אחת: לוג של 2, כלומר 1.",
      say="אפס הופעות: לוג של אחת, כלומר אפס. הופעה אחת: לוג של שתיים, כלומר אחת.", do=show("lt-0-1", "lt-0-2")),
    P("שלוש הופעות: לוג של 4, כלומר 2. ושבע: לוג של 8, כלומר 3.",
      say="שלוש הופעות: לוג של ארבע, כלומר שתיים. ושבע: לוג של שמונה, כלומר שלוש.", do=show("lt-0-3", "lt-0-4")),
    P("כדי להוסיף עוד 1 ל-TF, צריך להכפיל את אחת ועוד מספר ההופעות.",
      say="כדי להוסיף עוד אחת לטִי אֶף, צריך להכפיל את אחת ועוד מספר ההופעות.", do=show("v-c2"), hold=0.4),
], stage=W.stack(W.big("v-l", "$TF = \\log_2(1 + \\text{count})$", size=44), LOGT, gap=40),
   panel=st.card("v-c2", "rule", "🔑 כל הכפלה של 1 + count מוסיפה 1 בדיוק"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: 256 מסמכים, והמילה מופיעה ב-32 מהם. במסמך של 40 מילים, ה-TF-IDF שלה הוא 0.3.",
      say="שאלה בסגנון מבחן: מאתיים חמישים ושישה מסמכים, והמילה מופיעה בשלושים ושניים מהם. במסמך של ארבעים מילים, הטִי אֶף, אַיי דִּי אֶף שלה הוא אפס נקודה שלוש."),
    P("כמה פעמים היא מופיעה במסמך? הולכים אחורה, צעד אחר צעד."),
    P("IDF: לוג של 256 חלקי 32, לוג של 8, כלומר 3.",
      say="אַיי דִּי אֶף: לוג של מאתיים חמישים ושש חלקי שלושים ושתיים, לוג של שמונה, כלומר שלוש.", do=show("x-1")),
    P("לכן TF הוא 0.3 חלקי 3, כלומר 0.1. ו-0.1 מתוך 40 מילים זה 4 הופעות.",
      say="לכן טִי אֶף הוא אפס נקודה שלוש חלקי שלוש, כלומר אפס נקודה אחת. ואפס נקודה אחת מתוך ארבעים מילים זה ארבע הופעות.",
      do=show("x-2", "x-3", "q-ok-0")),
    P("12 יוצא מי שכפל את 0.3 ב-40 ושכח את ה-IDF.",
      say="שתים עשרה יוצא מי שכפל את אפס נקודה שלוש בארבעים ושכח את האַיי דִּי אֶף.", do=show("q-x-1", "q-n-1")),
    P("3 יוצא מי שהוסיף 1 ל-IDF, אבל השאלה לא נתנה את הנוסחה הזו.",
      say="שלוש יוצא מי שהוסיף אחת לאַיי דִּי אֶף, אבל השאלה לא נתנה את הנוסחה הזו.", do=show("q-x-2", "q-n-2")),
    P("ו-0.1 הוא ה-TF עצמו. זו המלכודת: עצרו באמצע, לפני שהכפילו באורך המסמך.",
      say="ואפס נקודה אחת הוא הטִי אֶף עצמו. זו המלכודת: עצרו באמצע, לפני שהכפילו באורך המסמך.",
      do=show("q-x-3", "q-n-3", "x-t"), hold=0.4),
], stage=W.stack(W.big("x-1", "$IDF = \\log_2 \\dfrac{256}{32} = 3$", size=44),
                 W.big("x-2", "$TF = \\dfrac{0.3}{3} = 0.1$", size=44),
                 W.big("x-3", "$\\text{count} = 0.1 \\times 40 = 4$", size=44),
                 st.card("x-t", "trap", "⚠️ בדרך אחורה: קודם IDF, אחר כך TF, ורק אז הופעות"), gap=26),
   panel='<div class="qstem"><span class="qid">בסגנון מבחן</span>256 מסמכים, המילה ב-32 מהם. במסמך של 40 מילים '
         'ה-TF-IDF שלה 0.3 (TF יחסי, IDF = log₂(N ⁄ df)). כמה פעמים היא מופיעה במסמך?</div>'
         + st.options("q", OPTS, notes={1: "בלי IDF", 2: "עם 1+", 3: "זה TF"}))

recap = Scene("recap", "סיכום", [
    P("לסיכום: TF כפול IDF. TF הוא ההופעות חלקי אורך המסמך.", say="לסיכום: טִי אֶף כפול אַיי דִּי אֶף. טִי אֶף הוא ההופעות חלקי אורך המסמך.",
      do=show("r-1")),
    P("IDF הוא לוג של מספר המסמכים חלקי מספר המסמכים שבהם המילה מופיעה.",
      say="אַיי דִּי אֶף הוא לוג של מספר המסמכים חלקי מספר המסמכים שבהם המילה מופיעה.", do=show("r-2")),
    P("ואם המילה בכל המסמכים, היחס הוא 1, והלוג של 1 הוא 0.", say="ואם המילה בכל המסמכים, היחס הוא אחת, והלוג של אחת הוא אפס.",
      do=show("r-3")),
    P("ובכל שאלה קוראים את הנוסחה: אולי יש 1+, ואולי TF לוגריתמי. נתראה ברמה הבאה.",
      say="ובכל שאלה קוראים את הנוסחה: אולי יש פלוס אחת, ואולי טִי אֶף לוגריתמי. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=TAB.replace('id="tf', 'id="tg').replace(' class="h"', ''),
   panel=st.card("r-1", "rule", "🔑 TF = הופעות ⁄ אורך")
         + st.card("r-2", "rule", "🔑 IDF = log₂(N ⁄ df)")
         + st.card("r-3", "rule", "🔑 log₂ 1 = 0")
         + st.card("r-4", "trap", "⚠️ קראו את הנוסחה: 1+? לוג על TF?"))

VIDEO = Video("U05", "unsup", "חישוב TF-IDF", [title, idea, calc, variants, logtf, exam, recap], bank={})
