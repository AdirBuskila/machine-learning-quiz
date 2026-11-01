# -*- coding: utf-8 -*-
"""unsup-course pronunciations (merged into lexicon.py for every video of the app). Only multi-letter English terms
that the U01-U08 subtitles print; the TTS gets the Hebrew spelling. Longest keys win (TF-IDF before TF)."""
WORDS = {
    "K-Means": "קֵיי מִינְס",
    "kmeans++": "קֵיי מִינְס פְּלָאס פְּלָאס",
    "TF-IDF": "טִי אֶף, אַיי דִּי אֶף",
    "IDF": "אַיי דִּי אֶף",
    "TF": "טִי אֶף",
    "WSSE": "דַּאבְּלְיוּ אֶס אֶס אִי",
    "Elbow": "אֶלְבּוֹאוּ",
    "PCA": "פִּי סִי אֵיי",
    "Bag of Words": "בֶּג אוֹף ווֹרְדְס",
    "Count Vectorizer": "קַאוּנְט וֶקְטוֹרַייזֶר",
    "Stop words": "סְטוֹפּ ווֹרְדְס",
    "stop words": "סְטוֹפּ ווֹרְדְס",
    "Tokenization": "טוֹקֶנִיזַצְיָה",
    "feature map": "פִיצֶ'ר מֶפּ",
    "Cell Histogram": "סֶל הִיסְטוֹגְרָמָה",
    "AdaBoost": "אָדָה בּוּסְט",
    "Random Forest": "רֶנְדוֹם פוֹרֶסְט",
    "bagging": "בֶּגִינְג",
    "boosting": "בּוּסְטִינְג",
    "MinMax": "מִין מַקְס",
    "RGB": "אַר גִ'י בִּי",
    "f1": "אֶף וָואן",
    "RMSE": "אַר אֶם אֶס אִי",
    "purity": "פְּיוּרִיטִי",
}

# single capital letters the unsup subtitles write in TeX: $K$ clusters, centroids $A$, $B$, $C$, a point $P$
TEX_TOKENS = {"K": "קֵיי", "A": "אֵיי", "B": "בִּי", "C": "סִי", "P": "פִּי"}
