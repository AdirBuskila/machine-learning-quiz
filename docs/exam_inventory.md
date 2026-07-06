# Exam inventory — Machine-Learning course

Source root: `C:\Users\Adir\Desktop\BSC\שנה ב\סמסטר ג\למידת מכונה`
Extraction: every exam/solutions/practice PDF was dumped with **both** pdfplumber and
PyMuPDF (fitz), plus a **bold/color-marked** dump (`*.marked.txt`) that preserves the
formatting text-extraction normally loses. DOCX via python-docx. See
`tools/raw/manifest.json` for per-file page/image/char metadata.

## How answers are keyed (verified by inspecting rendered pages)
- **Form-0 exams** (`מבחן מס' 000` / `טופס 0` / `form_0` in the header): the correct answer is
  the **first printed option** (א / 1). Verified on 26B, 25S, 24S-A.
- **Solutions files** (`פתרונות` / `עם תשובות` / `_sol`): the correct option is marked in
  **bold**; plain text drops bold, but fitz font-flags recover it (marked as `**…**` in
  `*.marked.txt`). Verified on 2022-בA, 2024b-C, 2023-summer, sample, practice sets.
- Some older solutions PDFs are set **entirely in a bold font** (e.g. 2022 סמ׳ א מועד ג) —
  bold can't discriminate the answer there, and many of their questions are figure/formula-
  dependent → **deferred** (see `ASK_ADIR.md`).

## Extraction-confidence legend
`HIGH` clean text + reliable key · `MED` usable but some garble/figure loss ·
`LOW/DEFER` scanned, all-bold, or figure-heavy → not auto-extracted, goes to `ASK_ADIR.md`.

## Exams by year

### 2026
| File | Format | Key | Form-0 | Confidence |
|---|---|---|---|---|
| `2026/ML 2026b moedA. form0.pdf` | text PDF | option 1 | **yes** | HIGH — clean, `מבחן מס' 000` |
| `2026/הערות סטודנט מועד א.pdf` | annotated scan | — | — | DEFER — plumber failed, 18pg annotations |

### 2025
| File | Format | Key | Form-0 | Confidence |
|---|---|---|---|---|
| `2025/למידת מכונה קיץ 2025 טופס 0.pdf` | text PDF | option 1 | **yes** | HIGH |
| `2025/למידת מכונה סמסטר ב 2025 מועד א.docx` | DOCX | none found | ? | DEFER — python-docx read 0 chars (content likely in text-boxes/images) |

### 2024  (each sitting ships a normal file **and** a "לא טופס 0" shuffled sibling — merged)
| Sitting | Question file | Answer source | Confidence |
|---|---|---|---|
| קיץ מועד א | `2024/סמסטר קיץ מועד א …` | ZIP `ML-2024c_moed_a_form_0.pdf` (**form-0 → opt 1**) | HIGH |
| קיץ מועד ב | `2024/סמסטר קיץ מועד ב …` | ZIP `ML-2024c_moed_b.pdf` (no key) | DEFER (no key) |
| קיץ מועד ג | `2024/סמסטר קיץ מועד ג …` | ZIP `ML-2024c_moed_c_sol.pdf` (**solutions**) | HIGH |
| סמ׳ ב מועד א | `2024/סמסטר ב מועד א 11.7.24.pdf` | ZIP `ML_2024b_moedA.pdf` — form-0? (auto-checked) | MED/DEFER |
| סמ׳ ב מועד ב | `2024/סמסטר ב מועד ב 4.8.24.pdf` | ZIP `ml_2024b_moedB.pdf` — form-0? (auto-checked) | MED/DEFER |
| סמ׳ ב מועד ג | `2024/סמסטר ב מועד ג 4.9.24.pdf` | ZIP `ml_2024b_moedC_sol.pdf` (**solutions**) | HIGH |
| מועד ב/ג אביב | `2024/מועד ב_/ג_ אביב 2024.pdf` | duplicates of סמ׳ ב files (identical char counts) | dedup |

### 2023
| Sitting | Answer source | Confidence |
|---|---|---|
| סמ׳ ב מועד א | `2023/סמסטר ב מועד א + פתרונות .pdf` (solutions) | MED-HIGH |
| סמ׳ ב מועד ב | `2023/סמסטר ב מועד ב + פתרונות.pdf` / `כולל פתרונות` | MED-HIGH |
| קיץ מועד א | `2023/…קיץ_מועד_א_עם_תשובות.pdf` / `+ פתרונות 19.10.23` | MED |
| מועד ג | `ZIP/2023/מועד ג 2023.docx` | DEFER (docx ~empty) |
| Ex10-Review | `ZIP/2023/Ex10-Review.pdf` | DEFER (very figure-heavy, 114 imgs) |

### 2022
| Sitting | Answer source | Confidence |
|---|---|---|
| סמ׳ א מועד א | ZIP `…2022א_מועד_א_עם_תשובות.pdf` (solutions) | MED (loose copies are scans) |
| סמ׳ א מועד ב | all copies **scanned** (≈128 chars/9pg) | DEFER — needs OCR |
| סמ׳ א מועד ג | `2022/…מועד ג …פתרונות…` | DEFER — all-bold + figure-heavy |
| סמ׳ ב מועד א | `2022/סמסטר ב מועד א פתרונות 2.6.pdf` (solutions) | HIGH — bold verified |
| סמ׳ ב מועד ב | `2022/סמסטר ב מועד ב + פתרונות 29.6.22.pdf` | MED-HIGH |
| סמ׳ ב מועד ג | `2022/סמסטר ב מועד ג פתרונות 14.8.22.pdf` | MED-HIGH |

## Sample exams (`מבחן לדוגמה/`)
| File | Key | Confidence |
|---|---|---|
| `מבחן לדוגמה פתרונות.pdf` (SAMP-1) | solutions (bold) | MED-HIGH |
| `מבחן לדוגמא עם פתרונות.pdf` (SAMP-2, 27pg) | solutions | MED |
| `שאלות לדוגמה + פתרונות.pdf` (SAMP-3, 27pg) | solutions | MED |
| `מבחן לדוגמא בלי פתרונות*.pdf` | none | skip (no key) |
| `תרגילים(2022).pdf` (69pg) | none | skip |
| `למידת מכונה - מבחנים דניאל.pdf` (45 MB) | — | SKIP — too large / aggregate |
| `קובץ עם מלא מבחנים.zip` (34 files) | mostly duplicates of loose exams + `2024c form_0`, `_sol` files, 2023 extras | mined for uniques |

## Practice sets (`קבצי תרגול למבחנים/`)
| File | Key | Confidence |
|---|---|---|
| `ניתוח טקסט ועיבוד תמונה - שאלות חזרה - תשובות.pdf` (PRAC-N) | solutions (bold) | HIGH — clean, modern, image-free |
| `שאלות לדוגמה - תשובות.pdf` (PRAC-X, 80pg) | solutions | MED (large) |
| `שאלות נוספות.pdf` | unclear | DEFER |
| `תרגילים של מירה המתרגלת.docx` + student-answer PDFs | student solutions | DEFER (verify manually) |

## Reference material (not exams — used for Learn mode)
- `מבחנים/דף נוסחאות.pdf`, `דף נוסחאות מורחב לזכאים.pdf` — formula sheets.
- `הרצאות + סיכומים/סיכומים/2025`, `2026`, `עד 2025/…` — per-topic summaries (richest).

## What I cannot reliably extract alone → `ASK_ADIR.md`
- Every **figure-dependent** question (confusion-matrix image, decision tree, scatter/
  regression plot, K-means diagram, image-processing picture, formula-as-image).
- **Scanned** 2022 sem-A moed-ב (needs OCR); the 2025 & 2023-מועד-ג **DOCX** (0-char reads).
- **All-bold** solutions PDFs where the correct option can't be told from the rest.
- 2024 sittings with **no answer key** (קיץ מועד ב; possibly סמ׳ ב מועד א/ב if not form-0).
