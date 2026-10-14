/* Evaluation course · level 4 — precision, recall, F1: computing them and reading the gap between them */
ML_LEVEL("eval", {
  id: "E04", order: 4, icon: "⚖️", title: "Precision, Recall ו-F1",
  goal: "לחשב precision, recall ו-F1 ממטריצה או מסיפור, ולפרש פער ביניהם במונחי FP ו-FN.",
  lesson: { html: `
<p>Accuracy מערבב את שתי המחלקות. שני המדדים הבאים מסתכלים רק על <strong>המחלקה החיובית</strong>, כל אחד מכיוון אחר:</p>
<div class="lesson-formula">Precision = TP / (TP + FP)</div>
<div class="lesson-formula">Recall = TP / (TP + FN)</div>
<table>
<thead><tr><th>מדד</th><th>המכנה</th><th>השאלה שהוא עונה עליה</th></tr></thead>
<tbody>
<tr><td><b>Precision</b> (דיוק)</td><td>כל מה שהמודל <b>אמר</b> שהוא חיובי (עמודה)</td><td>"מבין מה שסימנתי, כמה צדקתי?"</td></tr>
<tr><td><b>Recall</b> (כיסוי)</td><td>כל מה שבאמת חיובי (שורה)</td><td>"מבין מה שקיים, כמה תפסתי?"</td></tr>
</tbody></table>
<p><b>F1</b> מאחד את שניהם ל<b>ממוצע הרמוני</b>, שנמשך כלפי הערך הנמוך מביניהם:</p>
<div class="lesson-formula">F1 = 2 · P · R / (P + R)</div>
<div class="lesson-ex">התפוחים מהשלב הקודם: TP = 48, FN = 12, FP = 20, TN = 120<br>Precision = 48 / (48 + 20) = 48/68 ≈ <b>70.6%</b><br>Recall = 48 / (48 + 12) = 48/60 = <b>80%</b><br>F1 = 2 · 0.706 · 0.8 / (0.706 + 0.8) = <b>0.75</b><br>ולשם השוואה, Accuracy = 84%</div>
<div class="lesson-rule">🔑 <strong>PRecision נפגע מ-FP · REcall נפגע מ-FN.</strong> לכן: precision נמוך ← הרבה false positives, recall נמוך ← הרבה false negatives. ב-25B-A-Q7 (precision 0.7, recall 0.9) המודל "מזהה כמעט את כל החיוביים, אבל יש הרבה false positives", וב-26B-A-Q6 (precision 0.96, recall 0.6) הוא "מדייק במה שחזה, אבל יש הרבה false negatives".</div>
<h3>סט המספרים החוזר</h3>
<p>TP = 90, FN = 10, FP = 30, TN = 50 (או אותם יחסים כפול 1.5 או 0.3) הופיע בחמישה מועדים. התוצאות: accuracy ≈ <b>77.8%</b>, precision = <b>75%</b>, recall = <b>90%</b>. אם אתה רואה את היחסים האלה, אתה כבר יודע את המספרים.</p>
<div class="lesson-trap">⚠️ <strong>אותו סיפור, מדד אחר.</strong> 23B-A-Q10 ו-23B-B-Q10 זהות מילה במילה, אבל באחת התשובה היא precision ≈ 75% ובשנייה recall ≈ 90%. אל תענה מהזיכרון. קרא <b>איזה מדד מבקשים</b>.</div>
<div class="lesson-trap">⚠️ <strong>מספר נכון, שם שגוי.</strong> המסיחים הם תמיד מספרים אמיתיים מהמטריצה שהוצמדו למדד הלא נכון: ב-22B-A-Q11 recall = 90% אבל האופציה אומרת "accuracy כ-90%". חשב את כל המדדים, ואז התאם שם למספר.</div>
` },
  questions: [
    { id: "E04-Q1", question: "בדיקה לגילוי מחלה נוסתה על 200 אנשים: 40 חולים ו-160 בריאים. הבדיקה זיהתה 32 מהחולים, וסימנה בטעות 16 בריאים כחולים. (החיובי = חולה.) מה ה-precision?",
      options: ["≈ 66.7%", "80%", "88%", "≈ 94.7%"], answer: 0,
      recap: "Precision = TP / (TP + FP): מבין מה שסומן חיובי",
      explanation: "TP = 32, FN = 8, FP = 16, TN = 144. Precision = 32 / (32 + 16) = 32/48 ≈ **66.7%**: מתוך 48 שסומנו חולים, רק 32 באמת חולים. 80% = 32/40 הוא ה-**recall**, ו-88% = 176/200 הוא ה-accuracy. 94.7% = 144/152 הוא ה-precision של המחלקה השלילית (\"בריא\"), לא של החיובית." },
    { id: "E04-Q2", question: "מסנן ספאם נבדק על 1000 מיילים: 300 ספאם ו-700 תקינים. הוא תפס 240 מהספאם, וסימן בטעות 40 מיילים תקינים כספאם. (החיובי = ספאם.) מה ה-recall?",
      options: ["80%", "≈ 85.7%", "90%", "≈ 94.3%"], answer: 0,
      recap: "Recall = TP / (TP + FN): מבין כל החיוביים האמיתיים",
      explanation: "TP = 240, FN = 60, FP = 40, TN = 660. Recall = 240 / (240 + 60) = 240/300 = **80%**: המסנן תפס 80% מהספאם הקיים. 85.7% = 240/280 הוא ה-**precision**, 90% = 900/1000 הוא ה-accuracy, ו-94.3% = 660/700 הוא ה-recall של המחלקה השלילית." },
    { id: "E04-Q3", question: "למודל precision = 0.6 ו-recall = 0.9. מה ה-F1 שלו?",
      options: ["0.72", "0.75", "0.54", "0.9"], answer: 0,
      recap: "F1 = 2PR / (P + R), ממוצע הרמוני",
      explanation: "F1 = 2 · 0.6 · 0.9 / (0.6 + 0.9) = 1.08 / 1.5 = **0.72**. 0.75 הוא הממוצע **הרגיל** (החשבוני), אבל F1 הוא ממוצע **הרמוני** ונמשך לערך הנמוך. 0.54 הוא רק המכפלה P · R בלי הנרמול, ו-0.9 הוא ה-recall לבדו." },
    { id: "E04-Q4", question: "נתונה מטריצת הבלבול הבאה (שורות = ערך אמיתי, עמודות = חיזוי המודל). איזו טענה נכונה?",
      table: { head: ["", "חיזוי Positive", "חיזוי Negative"], rows: [["אמת Positive", "70", "30"], ["אמת Negative", "10", "90"]] },
      options: ["ה-precision הוא 87.5%", "ה-recall הוא 87.5%", "ה-accuracy הוא 70%", "מספר ה-False Negatives הוא 10"], answer: 0,
      recap: "שרטט TP/FN/FP/TN, חשב הכול, ואז התאם שם למספר",
      explanation: "TP = 70, FN = 30, FP = 10, TN = 90. Precision = 70/80 = **87.5%** ✓. ה-recall הוא 70/100 = 70%, כלומר 87.5% הוא מספר נכון עם שם שגוי. ה-accuracy הוא 160/200 = 80%, ו-70% הוא ה-recall. FN = 30, ו-10 הוא FP (מי שהפך שורות ועמודות)." },
    { id: "E04-Q5", question: "במודל סיווג, precision = 0.95 ו-recall = 0.4. מה המשמעות של הפער?",
      options: ["המודל כמעט תמיד צודק כשהוא חוזה חיובי, אבל יש הרבה false negatives", "המודל כמעט תמיד צודק כשהוא חוזה חיובי, אבל יש הרבה false positives", "המודל מזהה כמעט את כל החיוביים, אבל יש הרבה false positives", "המודל מזהה כמעט את כל החיוביים, אבל יש הרבה false negatives"], answer: 0,
      recap: "precision גבוה = מעט FP · recall נמוך = הרבה FN",
      explanation: "precision גבוה (0.95) ← כשהמודל אומר \"חיובי\" הוא כמעט תמיד צודק, כלומר **מעט FP**. recall נמוך (0.4) ← הוא תופס רק 40% מהחיוביים, כלומר **הרבה FN**. \"מזהה כמעט את כל החיוביים\" מתאר recall גבוה, וזה לא המצב כאן. זו התבנית של 26B-A-Q6." },
    { id: "E04-Q6", question: "במודל סיווג, precision = 0.5 ו-recall = 0.98. מה המשמעות של הפער?",
      options: ["המודל מזהה כמעט את כל החיוביים, אבל יש הרבה false positives", "המודל מזהה כמעט את כל החיוביים, אבל יש הרבה false negatives", "המודל כמעט תמיד צודק כשהוא חוזה חיובי, אבל יש הרבה false negatives", "המודל כמעט תמיד צודק כשהוא חוזה חיובי, אבל יש הרבה false positives"], answer: 0,
      recap: "recall גבוה = מעט FN · precision נמוך = הרבה FP",
      explanation: "recall 0.98 ← המודל **תופס כמעט את כל החיוביים** (מעט FN). precision 0.5 ← חצי ממה שהוא מסמן כחיובי שגוי, כלומר **הרבה FP**. החצי הראשון של התשובה מתאים למדד הגבוה, והחצי השני למדד הנמוך (25B-A-Q7)." },
    { id: "E04-Q7", question: "למודל יש TP = 50, FN = 10, FP = 10. כעת המודל מסמן כחיוביות עוד 20 דוגמאות שליליות (כלומר FP גדל ב-20, ושאר הערכים לא משתנים). מה קורה ל-precision ול-recall?",
      options: ["ה-precision יורד וה-recall לא משתנה", "ה-recall יורד וה-precision לא משתנה", "שניהם יורדים", "ה-precision עולה וה-recall יורד"], answer: 0,
      recap: "FP נמצא רק במכנה של precision",
      explanation: "FP מופיע **רק במכנה של precision**: הוא יורד מ-50/60 ≈ 83.3% ל-50/80 = 62.5%. ה-recall = TP / (TP + FN) = 50/60 בשני המקרים, כי FN ו-TP לא השתנו. PRecision ← FP, REcall ← FN." },
    { ref: "23B-A-Q10", recap: "TP = 90, FP = 30 ← precision = 90/120 = 75%" },
    { ref: "SAMP-1-Q3", recap: "TP = 400, FP = 100 ← precision = 400/500 = 0.8; TN = 500, לא 600" },
    { ref: "25B-A-Q7", recap: "recall גבוה + precision נמוך ← מזהה כמעט הכול, אבל הרבה FP" },
  ],
});
