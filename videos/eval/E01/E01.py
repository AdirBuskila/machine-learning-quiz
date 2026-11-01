# -*- coding: utf-8 -*-
"""E01 · מה זה למידת מכונה? — video script (build: tools/video/vid.py E01).

Source: courses/eval/E01.js (goal: supervised or not, regression or classification, binary / multi-class /
multi-label, and the input / output of training). Exam anchor: 25S-A-Q18 (hearing-aid noise, Van Gogh prices,
grouping articles, air pollution), which no guided quiz uses; no quiz item repeats its four problems.
Not used on purpose: every E01 quiz scenario (apartment price, film genres, storm strength, store customers, the
rain / fraud / hobbies / fruit set, ID3 "which is the model", the invoice check) and the quiz refs 24B-A-Q11
(score / all languages / honours / three road objects), 25B-A-Q16 and 23S-A-Q9; also the lesson's own examples.
Worked examples are new: bus arrival minutes, plant watering, 12 bird species, instruments in a recording, grouping
songs, VAT at a fixed rate, and a training run of linear regression whose learned line is computed below (numpy).
"""
import numpy as np

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

# ---------------------------------------------------------------- the training demo (verified)
KM = np.array([1.0, 2.0, 4.0, 5.0, 7.0])                 # distance to the stop, km
MIN = np.array([5.0, 7.0, 11.0, 13.0, 17.0])             # minutes until the bus arrived
b, a = np.polyfit(KM, MIN, 1)
assert abs(b - 2) < 1e-9 and abs(a - 3) < 1e-9           # the learned model: minutes = 2·km + 3
assert abs(2 * 6 + 3 - 15) < 1e-12                       # a new stop 6 km away -> 15 minutes

# my six problems: (text, kind, reason)
PROBS = [
    ("כמה דקות עד שהאוטובוס יגיע", "רגרסיה", "התווית היא מספר רציף"),
    ("האם הצמח צריך השקיה היום", "סיווג בינארי", "תווית אחת מתוך 2"),
    ("איזה מין ציפור בתמונה, מתוך 12 מינים", "multi-class", "תווית אחת מתוך 12"),
    ("<b>כל</b> הכלים שמנגנים בקטע מוזיקה", "multi-label", "כמה תוויות יחד"),
    ("קיבוץ שירים לפי דמיון, בלי שום תווית", "לא מונחית", "אין תווית: אשכול"),
    ("חישוב מע״מ לפי אחוז קבוע", "לא למידת מכונה", "נוסחה ידועה מראש"),
]

# ---------------------------------------------------------------- exam 25S-A-Q18 (verified by hand, listed here)
Q = BANK["25S-A-Q18"]
assert Q["correctIndex"] == 0 and "ואן גוך" in Q["question"] and "קיבוץ מאמרים" in Q["question"]
KIND = {1: ("סיווג בינארי", True), 2: ("רגרסיה", True), 3: ("אשכול, לא מונחית", False), 4: ("רגרסיה", True)}
assert [k for k, (_, sup) in KIND.items() if not sup] == [3]                   # only problem 3 has no label
assert [k for k, (t, _) in KIND.items() if "סיווג" in t] == [1]                 # only problem 1 is classification
OPTS = w.opts(Q)

XP = w.table("xp", ["בעיה", "תווית?", "סוג"],
             [["1 · רעש לסינון, או לא", "כן, אחת מ-2", "סיווג בינארי"],
              ["2 · שווי תמונה של ואן גוך", "כן, מספר", "רגרסיה"],
              ["3 · קיבוץ מאמרים לפי דמיון", "<b>אין</b>", "אשכול"],
              ["4 · רמת זיהום האוויר בערב", "כן, מספר", "רגרסיה"]], size=28)
PT = w.table("pt", ["הבעיה", "הסוג", "למה"], [[t, f"<b>{w.bidi(k)}</b>", r] for t, k, r in PROBS], size=27)

PIPE = ('<div class="ev-pipe">'
        '<div id="p-1" class="ev-node a h">קבוצת אימון<br><small>דוגמאות עם תווית</small></div>'
        '<div id="p-a1" class="ev-arr h">←</div>'
        '<div id="p-2" class="ev-node b h">אלגוריתם<br><small>רגרסיה לינארית</small></div>'
        '<div id="p-a2" class="ev-arr h">←</div>'
        '<div id="p-3" class="ev-node ok h">מודל<br>$\\hat{y} = 2x + 3$</div></div>')
DATA = w.table("dt", ["מרחק מהתחנה (ק״מ)", "דקות עד האוטובוס"],
               [[f"{int(x)}", f"{int(y)}"] for x, y in zip(KM, MIN)], hide_rows=False, size=28)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבין מה זה בכלל למידת מכונה, ומה נכנס ויוצא מאימון."),
    P("ואז נלמד לסווג כל בעיה בארבע שאלות, ונפתור שאלה ממבחן עם ארבע בעיות.", hold=0.4),
], stage='<div class="kicker">E01 · הערכת מודל</div><h1>מה זה למידת מכונה?</h1>'
         '<p>מונחית או לא, רגרסיה או סיווג, בינארי, multi-class או multi-label. ומה הקלט והפלט של האימון.</p>',
   layout="title")

train = Scene("train", "קלט ופלט של אימון", [
    P("בתכנות רגיל, מתכנת כותב את החוק. בלמידת מכונה, האלגוריתם מוצא את החוק לבד, מתוך **דוגמאות**.", do=show("dt")),
    P("כאן כל שורה היא דוגמה: כמה רחוק היה האוטובוס מהתחנה, והזמן שלקח לו להגיע. הזמן הוא ה**תווית**, מה שרוצים לחזות.",
      do=show("t-1")),
    P("את הדוגמאות האלה נותנים לאלגוריתם. זו **קבוצת האימון**.", do=show("p-1")),
    P("האלגוריתם הוא המתכון. כאן: רגרסיה לינארית, שמחפשת את הקו שהכי מתאים לנקודות.", do=show("p-a1", "p-2")),
    P("והתוצר הוא קו עם מספרים מסוימים: פעמיים המרחק, ועוד שלוש. זה ה**מודל**.", do=show("p-a2", "p-3")),
    P("עכשיו מגיעה תחנה חדשה, שישה קילומטרים מהתחנה. המודל עונה: חמש עשרה דקות.", do=show("t-2")),
    P("אז מה נכנס לאימון? דוגמאות. מה יוצא ממנו? מודל, שאפשר להפעיל על דוגמאות חדשות.", do=show("t-3"), hold=0.4),
], stage=w.col(DATA, PIPE, gap=30),
   panel=st.card("t-1", "plain", "דוגמה = מאפיינים + <b>תווית</b>")
         + st.card("t-2", "ex", "תחנה חדשה: $x = 6$ ← $\\hat{y} = 2 \\cdot 6 + 3 = 15$ דקות")
         + st.card("t-3", "rule", "🔑 נכנס: <b>קבוצת אימון</b> · יוצא: <b>מודל</b>. האלגוריתם הוא הדרך, לא התוצר"))

FLOW = [
    ("q1", "1. אין תווית, רק קבוצות? ← <b>לא מונחית</b>"),
    ("q2", "2. הכלל ידוע מראש? ← <b>לא למידת מכונה</b>"),
    ("q3", "3. התווית מספר רציף? ← <b>רגרסיה</b>"),
    ("q4", "4. התווית קטגוריה? ← <b>סיווג</b>"),
]
kinds = Scene("kinds", "ארבע שאלות", [
    P("כדי לסווג בעיה שואלים ארבע שאלות, לפי הסדר. ננסה אותן על שש בעיות חדשות.", do=show("q1", "q2", "q3", "q4")),
    P("כמה דקות עד שהאוטובוס יגיע? יש תווית, והיא מספר רציף: **רגרסיה**.", do=show("pt-r0")),
    P("האם הצמח צריך השקיה היום? התווית היא כן או לא, אחת משתיים: **סיווג בינארי**.", do=show("pt-r1")),
    P("איזה מין ציפור בתמונה, מתוך שנים עשר מינים? תווית **אחת** מתוך הרבה: **multi-class**.", do=show("pt-r2"),
      say="איזה מין ציפור בתמונה, מתוך שנים עשר מינים? תווית אחת מתוך הרבה: מוּלְטִי קְלַאס."),
    P("ואילו **כל** הכלים שמנגנים בקטע? לקטע אחד יש כמה תוויות בבת אחת: **multi-label**.", do=show("pt-r3"),
      say="ואילו כל הכלים שמנגנים בקטע? לקטע אחד יש כמה תוויות בבת אחת: מוּלְטִי לֵייבֶּל."),
    P("קיבוץ שירים לפי דמיון, כשאין לאף שיר תווית? אין מה לחזות, רק לגלות קבוצות: **לא מונחית**.", do=show("pt-r4")),
    P("וחישוב מע״מ לפי אחוז קבוע? הנוסחה ידועה מראש. אין כאן מה ללמוד.", do=show("pt-r5"), say="וחישוב מַעַם לפי אחוז קבוע? הנוסחה ידועה מראש. אין כאן מה ללמוד."),
    P("שימו לב: מונחית מול לא מונחית, ורגרסיה מול סיווג, הם **שני צירים שונים**.", do=show("k-t")),
    P("השאלה הראשונה היא אם יש תווית. רק אחרי שיש, שואלים מאיזה סוג היא.", hold=0.4),
], stage=w.col(PT),
   panel="".join(st.card(i, "plain" if i != "q4" else "plain", t) for i, t in FLOW)
         + st.card("k-t", "trap", "⚠️ יש תווית? ← מונחית או לא. איזו תווית? ← רגרסיה או סיווג"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: ארבע בעיות, ואיזו טענה עליהן נכונה. נעבור על כל בעיה בארבע השאלות.", do=show("xp")),
    P("בעיה אחת: האם האות הוא רעש לסינון, או לא. תווית אחת מתוך שתיים: סיווג בינארי.", do=show("xp-r0")),
    P("בעיה שתיים: כמה שווה תמונה של ואן גוך. התווית היא מחיר, מספר: רגרסיה.", do=show("xp-r1")),
    P("בעיה שלוש: לקבץ מאמרים לפי הדמיון ביניהם. אין שום תווית: אשכול, למידה לא מונחית.", do=show("xp-r2")),
    P("בעיה ארבע: רמת זיהום האוויר בסוף היום, ערך מספרי: רגרסיה.", do=show("xp-r3"), hold=0.3),
    P("עכשיו הטענות. כל ארבע הבעיות הן למידת מכונה, ורק השלישית היא אשכול. זו הטענה הנכונה.", do=show("q-ok-0")),
    P("״כולן מונחות״ נופל על הבעיה השלישית: אין בה תווית. וזה פוסל שתי טענות.",
      do=show("q-x-1", "q-n-1", "q-x-3", "q-n-3")),
    P("שימו לב שבאחת מהן החצי השני נכון: באמת רק הראשונה היא סיווג. אבל טענה נכונה רק אם **כולה** נכונה.",
      do=show("x-t")),
    P("והטענה שהשנייה אינה למידת מכונה כי היא רגרסיה? רגרסיה היא בדיוק למידת מכונה מונחית.",
      do=show("q-x-2", "q-n-2"), hold=0.6),
], stage=w.col(XP, st.card("x-t", "trap", "⚠️ טענה עם ״אך״: שני החלקים חייבים להיות נכונים"), gap=28),
   panel=w.stem(Q["id"], "ארבע בעיות למידה. איזו טענה נכונה?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={1: "3 לא מונחית", 2: "רגרסיה היא ML",
                                                                    3: "3 לא מונחית"}) + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: אימון מקבל קבוצת אימון ומוציא מודל. האלגוריתם הוא הדרך.", do=show("r-1")),
    P("אין תווית, רק קבוצות? לא מונחית. כלל שידוע מראש? בכלל לא למידת מכונה.", do=show("r-2")),
    P("תווית מספרית? רגרסיה. תווית קטגוריה? סיווג.", do=show("r-3")),
    P("ובסיווג סופרים: אחת משתיים, אחת מכמה, או כמה יחד. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(w.table("rc", ["כמה תוויות לדוגמה?", "מתוך כמה?", "סוג"],
                       [["אחת", "2", "בינארי"], ["אחת", "3 ומעלה", w.bidi("multi-class")],
                        ["כמה יחד", "—", w.bidi("multi-label")]], hide_rows=False, size=32)),
   panel=st.card("r-1", "rule", "🔑 קבוצת אימון ← <b>אימון</b> ← מודל")
         + st.card("r-2", "rule", "🔑 אין תווית ← לא מונחית · כלל ידוע ← לא <span class='ev-l'>ML</span>")
         + st.card("r-3", "rule", "🔑 מספר ← רגרסיה · קטגוריה ← סיווג")
         + st.card("r-4", "trap", "⚠️ <b>״כל ה...״</b> ← כמה תוויות יחד"))

VIDEO = w.check_say(Video("E01", "eval", "מה זה למידת מכונה?", [title, train, kinds, exam, recap],
                          bank={"25S-A-Q18": 0}))
