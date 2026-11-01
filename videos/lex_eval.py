# -*- coding: utf-8 -*-
"""Lexicon additions for the evaluation-course videos (E01..E10). Only distinctive metric / method names that appear
in subtitles. WORDS is plain substring replacement over every video of the app, so no short or common English word
is listed (train, test, fold, set): the scripts give an explicit say= for those phrases, and w_eval.check_say()
refuses any Latin letter left in the spoken text."""

WORDS = {
    "Precision": "פְּרֶסִיזְ'ן",
    "precision": "פְּרֶסִיזְ'ן",
    "Recall": "רִיקוֹל",
    "recall": "רִיקוֹל",
    "Accuracy": "אַקְיוּרַסִי",
    "accuracy": "אַקְיוּרַסִי",
    "error rate": "אֶרוֹר רֵייט",
    "overfitting": "אוֹבֶרְפִיטִינְג",
    "underfitting": "אַנְדֶרְפִיטִינְג",
    "multi-class": "מוּלְטִי קְלַאס",
    "multi-label": "מוּלְטִי לֵייבֶּל",
    "macro": "מַקְרוֹ",
    "micro": "מַיְקְרוֹ",
    "hold-out": "הוֹלְד אַאוּט",
    "K-fold": "קֵיי פוֹלְד",
    "Leave-One-Out": "לִיב וַואן אַאוּט",
    "min-max": "מִין מַקְס",
    "z-score": "זֶד סְקוֹר",
    "PCA": "פִּי סִי אֵיי",
    "K-Means": "קֵיי מִינְס",
    "KNN": "קֵיי אֶן אֶן",
}

HOMOPHONES = {}

TEX_TOKENS = {
    "TP": "טִי פִּי", "FP": "אֶף פִּי", "FN": "אֶף אֶן", "TN": "טִי אֶן",
}
