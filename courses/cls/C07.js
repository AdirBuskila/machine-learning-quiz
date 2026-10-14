/* Classifiers course · level 7 — Naive Bayes by hand: prior, likelihood from a table, posterior, the direction of the bar */
ML_LEVEL("cls", {
  id: "C07", order: 7, icon: "🧮", title: "נאיב בייס: חישוב מטבלה",
  goal: "לחשב prior ו-likelihood מטבלה, להכפיל ל-posterior ולסווג, ולזהות את המלכודות של המכנה, הפס האנכי וה-prior החסר.",
  lesson: { html: `
<p><strong>נאיב בייס</strong> מסווג דוגמה חדשה X כך: לכל מחלקה Y הוא שואל "כמה סביר ש-X שייכת ל-Y?", ובוחר במחלקה עם הציון הגבוה. שלושה מונחים, והמבחן מבלבל ביניהם בכוונה:</p>
<table>
<thead><tr><th>מונח</th><th>סימון</th><th>מה זה</th></tr></thead>
<tbody>
<tr><td><b>Prior</b></td><td>P(Y)</td><td>כמה המחלקה שכיחה ב-train, <b>לפני</b> שראינו את הדוגמה</td></tr>
<tr><td><b>Likelihood</b></td><td>P(X | Y)</td><td>בתוך המחלקה Y, כמה שכיח הערך X</td></tr>
<tr><td><b>Posterior</b></td><td>P(Y | X)</td><td>מה שמשווים בסוף: likelihood × prior</td></tr>
</tbody></table>
<div class="lesson-formula">P(Y | X) ∝ P(X | Y) · P(Y)</div>
<p>הסימן ∝ אומר "פרופורציונלי": בחוק בייס המלא מחלקים גם ב-P(X), אבל זה <b>אותו מכנה לכל המחלקות</b>, ולכן הוא לא משנה מי מנצח ואף פעם לא צריך אותו (22B-C-Q1). עם כמה מאפיינים, ה-likelihood הוא <b>מכפלה</b>: P(x₁|Y) · P(x₂|Y) · …</p>
<div class="lesson-ex"><b>דוגמה:</b> 6 שורות (גיל, עניבה): (בינוני, No), (צעיר, No), (מבוגר, No), (מבוגר, No), (מבוגר, Yes), (צעיר, Yes). דוגמה חדשה: מבוגר.<br>
<b>prior:</b> Yes = 2/6, No = 4/6.<br>
<b>likelihood:</b> בתוך Yes (2 שורות) יש מבוגר אחד → <b>½</b>. בתוך No (4 שורות) יש 2 מבוגרים → 2/4 = <b>½</b>.<br>
ה-likelihood שווה, ולכן מכריעים לפי ה-prior → <b>No</b>. זו בדיוק 23S-A-Q14.</div>
<div class="lesson-rule">🔑 <strong>המכנה של likelihood הוא גודל המחלקה, לא כל הטבלה.</strong> ספור קודם כמה שורות יש בכל מחלקה, ורק בתוכן חפש את הערך.</div>
<div class="lesson-trap">⚠️ <strong>0.333 ו-0.666 הם לא likelihood.</strong> אם סופרים בין 3 השורות של "מבוגר" כמה מכל מחלקה, מקבלים ⅓ ו-⅔, וזה P(Y | X), כלומר posterior. ב-23S-A-Q14 וב-SAMP-1-Q8 זה המסיח, יחד עם "נסווג אקראית": likelihood שווה לא אומר אקראי, יש prior.</div>
<div class="lesson-trap">⚠️ <strong>prior חסר וזוג מוצלב.</strong> ב-26B-A-Q16 נתון רק P(not-MSc) = 0.7, ולכן P(MSc) = 1 − 0.7 = 0.3.<br>עבור MSc: ‏0.6 · 0.3 = <b>0.18</b><br>עבור not-MSc: ‏0.3 · 0.7 = <b>0.21</b><br>המסיח 0.42 = 0.6 · 0.7 מכפיל likelihood של מחלקה אחת ב-prior של <b>השנייה</b>.</div>
<div class="lesson-trap">⚠️ <strong>כיוון הפס האנכי.</strong> ב-25B-A-Q9 נתונים שני דברים שונים:<br>P(X | outstanding) = 0.5 הוא likelihood, ולכן מכפילים ב-prior: ‏0.5 · 0.2 = <b>0.1</b><br>P(not-outstanding | X) = 0.2 הוא <b>כבר posterior</b>, ולא מכפילים אותו שוב.<br>‏0.2 גדול מ-0.1, ולכן לא מצטיין. משווים posteriors, לא likelihoods.</div>
` },
  questions: [
    { id: "C07-Q1", question: "מאמנים נאיב בייס על הטבלה. מה ה-**likelihood** של הדוגמה החדשה color = red עבור כל מחלקה?",
      table: { head: ["color", "buy"], rows: [["red", "Yes"], ["red", "Yes"], ["blue", "Yes"], ["green", "Yes"], ["red", "No"], ["blue", "No"], ["blue", "No"], ["green", "No"]] },
      options: ["P(red|Yes) = 0.5, P(red|No) = 0.25", "P(red|Yes) ≈ 0.667, P(red|No) ≈ 0.333", "P(red|Yes) = 0.25, P(red|No) = 0.125", "P(red|Yes) = 0.375, P(red|No) = 0.375"], answer: 0,
      recap: "likelihood = ספירה בתוך המחלקה ⁄ גודל המחלקה",
      explanation: "במחלקה Yes יש 4 שורות, ומהן 2 עם red, ולכן 2/4 = **0.5**. במחלקה No יש 4 שורות, ומהן אחת עם red, ולכן 1/4 = **0.25**. ⅔ ו-⅓ הן החלוקה של 3 השורות עם red בין המחלקות, כלומר posterior. 0.25 ו-0.125 יוצאים מחלוקה בכל 8 השורות במקום בגודל המחלקה, ו-0.375 הוא סתם 3/8 שורות red בטבלה." },
    { id: "C07-Q2", question: "באותה טבלה, איך תסווג הדוגמה color = red?",
      table: { head: ["color", "buy"], rows: [["red", "Yes"], ["red", "Yes"], ["blue", "Yes"], ["green", "Yes"], ["red", "No"], ["blue", "No"], ["blue", "No"], ["green", "No"]] },
      options: ["Yes, כי 0.5 · 0.5 = 0.25 גדול מ-0.25 · 0.5 = 0.125", "No, כי ל-No יש יותר שורות blue", "אי אפשר לסווג בלי P(red)", "אקראית, כי ה-prior של שתי המחלקות שווה"], answer: 0,
      recap: "posterior ∝ likelihood × prior, ומשווים בין המחלקות",
      explanation: "ה-prior של כל מחלקה 4/8 = 0.5. עבור Yes: ‏0.5 · 0.5 = **0.25**. עבור No: ‏0.25 · 0.5 = 0.125. לכן **Yes**. prior שווה לא אומר אקראי: הוא פשוט מתבטל, וה-likelihood מכריע. ו-P(red) הוא אותו מכנה לשתי המחלקות, ולכן אין בו צורך." },
    { ref: "23S-A-Q14", recap: "likelihood שווה (½ מול 2/4) → מכריעים לפי ה-prior, לא אקראית" },
    { id: "C07-Q3", question: "ב-train יש 3 דוגמאות ממחלקה A, ומהן אחת עם x = 1. יש 6 דוגמאות ממחלקה B, ומהן 2 עם x = 1. הגיעה דוגמה עם x = 1. מה נכון?",
      options: ["ה-likelihood הוא ⅓ לשתי המחלקות, ולכן נכריע לפי ה-prior ונסווג B", "ה-likelihood הוא ⅓ לשתי המחלקות, ולכן נסווג אקראית", "ה-likelihood הוא ⅓ ל-A ו-⅔ ל-B, ולכן נסווג B", "ה-likelihood הוא ⅓ ל-A ו-⅔ ל-B, ולכן נסווג A"], answer: 0,
      recap: "likelihood שווה מתבטל, ומה שנשאר הוא ה-prior",
      explanation: "A: 1/3. B: 2/6 = 1/3. **שווה**, ולכן מתבטל, ומכריע ה-prior: B = 6/9 מול A = 3/9 → **B**. ⅓ ו-⅔ הם החלוקה של שלוש הדוגמאות עם x = 1 בין המחלקות, כלומר posterior ולא likelihood. ו\"אקראית\" שגוי: יש prior להכריע לפיו." },
    { id: "C07-Q4", question: "מסנן ספאם (נאיב בייס) נותן עבור מייל חדש X: ‏P(X | spam) = 0.3 ו-P(X | ham) = 0.05, וה-prior של ham הוא 0.8. איך יסווג המייל ובאיזה ציון (בלי ה-prior של הדוגמה)?",
      options: ["spam, עם 0.06", "spam, עם 0.24", "ham, עם 0.04", "ham, עם 0.05"], answer: 0,
      recap: "prior חסר = 1 פחות השני. מכפילים likelihood ו-prior של אותה מחלקה",
      explanation: "P(spam) = 1 − 0.8 = 0.2. spam: 0.3 · 0.2 = **0.06**. ham: 0.05 · 0.8 = 0.04. ‏0.06 > 0.04 → **spam**. 0.24 = 0.3 · 0.8 מכפיל את ה-likelihood של spam ב-prior של ham, הזוג המוצלב של 26B-A-Q16. 0.05 הוא likelihood בלבד, בלי prior." },
    { ref: "26B-A-Q16", recap: "P(MSc) = 1 − 0.7 = 0.3 → MSc 0.18 מול not-MSc 0.21" },
    { id: "C07-Q5", question: "מודל נאיב בייס נותן עבור דוגמה X: ‏P(X | A) = 0.4 וגם P(B | X) = 0.3. ה-prior של A הוא 0.5 (ויש רק שתי מחלקות). מה משווים ולאיזו מחלקה נסווג (בלי ה-prior של הדוגמה)?",
      options: ["משווים posteriors: A = 0.2 מול B = 0.3, ולכן B", "משווים posteriors: A = 0.2 מול B = 0.15, ולכן A", "משווים likelihoods: A = 0.4 מול B = 0.3, ולכן A", "משווים likelihoods: A = 0.2 מול B = 0.3, ולכן B"], answer: 0,
      recap: "P(X|Y) צריך הכפלה ב-prior · P(Y|X) כבר מוכן",
      explanation: "P(X | A) הוא likelihood: 0.4 · 0.5 = **0.2**. P(B | X) הוא **כבר posterior** (המחלקה משמאל לפס), ולכן 0.3 כמו שהוא. ‏0.3 > 0.2 → **B**. ‏0.15 יוצא אם מכפילים את 0.3 שוב ב-prior, ו\"משווים likelihoods\" שגוי: בסוף תמיד משווים posteriors. זו התבנית של 25B-A-Q9." },
    { ref: "25B-A-Q9", recap: "pr(not-outstanding|X) = 0.2 כבר posterior; outstanding = 0.5 · 0.2 = 0.1" },
    { id: "C07-Q6", question: "נאיב בייס עם שני מאפיינים. P(Y) = 0.6 ו-P(x₁|Y) = 0.5 ו-P(x₂|Y) = 0.4. ‏P(N) = 0.4 ו-P(x₁|N) = 0.3 ו-P(x₂|N) = 0.9. לאיזו מחלקה תסווג הדוגמה (x₁, x₂)?",
      options: ["Y, כי 0.12 > 0.108", "N, כי 0.27 > 0.2", "N, כי 0.9 הוא ה-likelihood הגבוה ביותר", "Y, כי ה-prior שלה גבוה יותר, ולכן אין צורך לחשב"], answer: 0,
      recap: "likelihood של כמה מאפיינים = מכפלה, ואז × prior",
      explanation: "Y: 0.6 · 0.5 · 0.4 = **0.12**. N: 0.4 · 0.3 · 0.9 = **0.108**. → **Y**, בהפרש קטן. 0.2 ו-0.27 הן המכפלות **בלי** ה-prior, ושם N מנצח, כך שהשמטת ה-prior הופכת את התשובה. וגם \"ה-prior גבוה, אין צורך לחשב\" שגוי: ה-likelihood יכול להפוך אותו." },
    { id: "C07-Q7", question: "נאיב בייס בינארי: P(x=1 | Y=0) = 0.75 ו-P(x=1 | Y=1) = 0.6 ו-P(Y=0) = P(Y=1) = 0.5. למה **לא** צריך את P(x=1) כדי לסווג דוגמה עם x = 1?",
      options: ["כי זה אותו מכנה לשתי המחלקות, והוא לא משנה מי גדול יותר", "כי P(x=1) תמיד שווה 1", "כי בנאיב בייס מניחים ש-x בלתי תלוי במחלקה", "צריך אותו, ובלעדיו אי אפשר לסווג"], answer: 0,
      recap: "P(X) מתבטל בהשוואה, אף פעם לא צריך אותו",
      explanation: "בחוק בייס המלא כל posterior מחולק ב-P(x=1), אבל זה **אותו מספר** לשתי המחלקות, ולכן ההשוואה 0.75 · 0.5 = 0.375 מול 0.6 · 0.5 = 0.3 לא משתנה → Y=0. האופציה \"חסרות ההסתברויות P(X)\" היא המסיח של 22B-C-Q1. אי-התלות בנאיב בייס היא בין **המאפיינים**, לא בין מאפיין למחלקה." },
  ],
});
