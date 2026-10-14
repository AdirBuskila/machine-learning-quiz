/* Evaluation course · level 6 — which metric for which problem: imbalanced data, F1, regression metrics (MAE/RMSE/R²), clustering metrics */
ML_LEVEL("eval", {
  id: "E06", order: 6, icon: "📐", title: "איזה מדד מתאים?",
  goal: "להתאים מדד לסוג הבעיה, לחשב MAE, לדעת מתי נמוך ומתי גבוה זה טוב, ולזהות מתי accuracy מטעה.",
  lesson: { html: `
<h3>כל סוג בעיה והמדדים שלו</h3>
<table>
<thead><tr><th>סוג הבעיה</th><th>מדדים</th></tr></thead>
<tbody>
<tr><td><b>סיווג</b></td><td>confusion matrix, accuracy, precision, recall, F1</td></tr>
<tr><td><b>רגרסיה</b></td><td>MAE, MSE, RMSE, R²</td></tr>
<tr><td><b>אשכול</b> (לא מונחית)</td><td>purity, NMI (Normalized Mutual Information)</td></tr>
</tbody></table>
<div class="lesson-trap">⚠️ <strong>שאלות עם הנחה כוזבת.</strong> SAMP-3-Q16 שואלת "מדוע נהוג להשתמש ב-purity למודל רגרסיה?", והתשובה היא ש-purity <b>אינו</b> מדד לרגרסיה. ב-25S-A-Q13 מטריצת בלבול לא רלוונטית ל-K-Means כי הוא אלגוריתם אשכול, לא מסווג. ב-SAMP-2-Q3 המדד הנפוץ לסיווג הוא confusion matrix, ו-MSE ו-NMI שייכים לסוגים אחרים.</div>
<h3>מדדי רגרסיה</h3>
<p>ברגרסיה התשובה היא מספר, ולכן מודדים <b>כמה רחוק</b> החיזוי ŷ מהאמת y:</p>
<div class="lesson-formula">MAE = Σ|y − ŷ| / n</div>
<div class="lesson-formula">MSE = Σ(y − ŷ)² / n</div>
<div class="lesson-formula">RMSE = √MSE</div>
<div class="lesson-ex">אמת: 5, 7, 10 · חיזוי: 6, 7, 8<br>השגיאות בערך מוחלט: 1, 0, 2<br>MAE = 3/3 = <b>1</b><br>MSE = (1 + 0 + 4)/3 ≈ <b>1.67</b><br>RMSE = √1.67 ≈ <b>1.29</b></div>
<div class="lesson-rule">🔑 <strong>MAE, MSE, RMSE הם מדדי שגיאה: נמוך יותר = טוב יותר.</strong> R² ו-accuracy: גבוה יותר = טוב יותר. בשאלת "מי בנה מודל טוב יותר" (23S-A-Q18, 24B-A-Q16, SAMP-1-Q22, 26S-A-Q8) התשובה תמיד: <b>בעל הציון הנמוך ביותר, בהנחה שהמדד הוא RMSE (או MAE)</b>.</div>
<div class="lesson-trap">⚠️ <strong>MAE לעולם אינו שלילי</strong> (ערך מוחלט), ו-MAE הוא <b>ממוצע</b>, לא השגיאה המקסימלית. ב-25B-A-Q3 התשובה MAE = 0.1 (במיליוני דולרים = טעות ממוצעת של 100,000$), והמסיחים הם 0.1−, 0.0333− ו"טעות מקסימלית". ו-accuracy <b>אינו</b> מדד רגרסיה, אז "בהנחה שהמדד הוא accuracy" נפסל מיד.</div>
<h3>נתונים לא מאוזנים</h3>
<p>dataset <b>לא מאוזן</b> = מחלקה אחת נדירה מאוד. ב-25B-A-Q20 תקלת מנוע קורית ב-1% מהמקרים: מודל שתמיד אומר "אין תקלה" מקבל <b>99% accuracy</b> ולא מוצא אף תקלה (recall = 0).</p>
<div class="lesson-rule">🔑 <strong>dataset לא מאוזן ← precision, recall או F1, לא accuracy</strong>. ה-F1 עדיף על accuracy "כי הוא מאזן בין precision ל-recall ומתמקד במחלקה החשובה" (25S-A-Q3).</div>
` },
  questions: [
    { id: "E06-Q1", question: "מודל רגרסיה חזה 22, 27, 40 עבור דוגמאות שערכיהן האמיתיים הם 20, 30, 40. מה ה-MAE?",
      options: ["≈ 1.67", "≈ −0.33", "3", "≈ 4.33"], answer: 0,
      recap: "MAE = ממוצע השגיאות בערך מוחלט",
      explanation: "השגיאות בערך מוחלט: |20−22| = 2, |30−27| = 3, |40−40| = 0. MAE = 5/3 ≈ **1.67**. −0.33 יוצא אם שוכחים את הערך המוחלט (2 ו-3 מקזזים זה את זה), ו-MAE לעולם אינו שלילי. 3 היא השגיאה **המקסימלית**, ו-4.33 = (4 + 9 + 0)/3 הוא ה-MSE." },
    { id: "E06-Q2", question: "ארבעה סטודנטים אימנו מודלי רגרסיה על אותם train-set ו-test-set. הציונים על ה-test-set: דנה 0.62, יואב 0.71, מיכל 0.80, עומר 0.93. בכל תשובה, הניחו שכולם השתמשו במדד שמוזכר בה. מי בנה את המודל הטוב ביותר?",
      options: ["עומר, בהנחה שהמדד הוא R²", "דנה, בהנחה שהמדד הוא R²", "עומר, בהנחה שהמדד הוא MAE", "מיכל, בהנחה שהמדד הוא accuracy"], answer: 0,
      recap: "R² גבוה = טוב · MAE/RMSE נמוך = טוב · accuracy אינו מדד רגרסיה",
      explanation: "ב-**R²** גבוה יותר = טוב יותר ← **עומר** (0.93) ✓. דנה הייתה מנצחת רק אם המדד היה מדד שגיאה, ו-R² אינו כזה. עם MAE (שגיאה) הציון הגבוה הוא הגרוע ביותר, כלומר עומר היה האחרון. accuracy אינו מדד לרגרסיה בכלל (המסיח של SAMP-1-Q22)." },
    { id: "E06-Q3", question: "מתוך 1000 עסקאות אשראי, 10 הן הונאה. מודל אחד פשוט מסווג **כל** עסקה כ\"לא הונאה\". (החיובי = הונאה.) מה ה-accuracy וה-recall שלו?",
      options: ["accuracy = 99%, recall = 0", "accuracy = 99%, recall = 99%", "accuracy = 1%, recall = 0", "accuracy = 99%, recall = 100%"], answer: 0,
      recap: "מחלקה נדירה ← accuracy גבוה מטעה; ה-recall חושף את הבעיה",
      explanation: "המודל צודק על 990 העסקאות התקינות ← accuracy = 990/1000 = **99%**. אבל הוא לא תפס אף הונאה: TP = 0, FN = 10 ← **recall = 0/10 = 0**. מכאן שבנתונים לא מאוזנים accuracy מטעה, והמסקנה של 25B-A-Q20: להשתמש ב-precision, recall או F1." },
    { id: "E06-Q4", question: "איזה מהמדדים הבאים מתאים להערכת מודל **אשכול** (clustering)?",
      options: ["Purity", "RMSE", "Recall", "R²"], answer: 0,
      recap: "אשכול ← purity / NMI · רגרסיה ← RMSE, R² · סיווג ← recall",
      explanation: "**Purity** (וגם NMI) מודד עד כמה כל אשכול \"טהור\", כלומר מכיל דוגמאות מקבוצה אחת, ולכן הוא מדד אשכול. RMSE ו-R² הם מדדי **רגרסיה**, ו-recall הוא מדד **סיווג** (SAMP-2-Q3, SAMP-3-Q16)." },
    { id: "E06-Q5", question: "איזה מהבאים **אינו** מדד להערכת מודל רגרסיה?",
      options: ["Accuracy", "MAE", "RMSE", "R²"], answer: 0,
      recap: "accuracy שייך לסיווג",
      explanation: "**Accuracy** סופר כמה חיזויים \"צדקו\", וזה מוגדר רק כשהתשובה היא קטגוריה ← סיווג. ברגרסיה החיזוי כמעט אף פעם לא שווה בדיוק לאמת, ולכן מודדים מרחק: MAE, RMSE, ואת R²." },
    { id: "E06-Q6", question: "בבעיית סיווג בינארית, מדוע F1-score עדיף לעיתים על accuracy?",
      options: ["כי הוא משלב את precision ואת recall של המחלקה החשובה, ולא מתנפח בגלל מחלקה גדולה וקלה", "כי הוא מהיר יותר לחישוב", "כי הוא תמיד גבוה יותר מה-accuracy", "כי הוא נותן משקל שווה לחיזוי המחלקה החיובית ולחיזוי המחלקה השלילית"], answer: 0,
      recap: "F1 = ממוצע הרמוני של P ו-R, ממוקד במחלקה החשובה",
      explanation: "F1 בנוי רק מ-precision ומ-recall של **המחלקה החיובית** (החשובה), ולכן TN רבים של מחלקה גדולה לא מנפחים אותו כמו את ה-accuracy (25S-A-Q3). \"מהיר יותר\" לא רלוונטי, F1 לא תמיד גבוה יותר, ו\"משקל שווה לשתי המחלקות\" הוא בדיוק מה ש-F1 **לא** עושה." },
    { ref: "25B-A-Q20", recap: "תקלה ב-1% ← לא מאוזן ← precision / recall / F1, לא accuracy" },
    { ref: "23S-A-Q18", recap: "מי טוב יותר? הציון הנמוך ביותר, בהנחה שהמדד הוא RMSE" },
    { ref: "25B-A-Q3", recap: "MAE = (0.1 + 0.2 + 0)/3 = 0.1, ממוצע ולא שלילי" },
  ],
});
