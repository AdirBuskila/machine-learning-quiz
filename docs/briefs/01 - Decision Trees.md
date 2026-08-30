# 01 — עצי החלטה (Decision Trees)

**39 questions** · app topic: **עצי החלטה** · 🔴 Gap 1 · Calendar: Aug 24–25
**Deep guide for the arithmetic:** [[Gap-1-Entropy-Exam-Recipe]] · **Index:** [[00 - Index]]
**תיאוריה בעברית (כל ה-28 השאלות התיאורטיות):** [[01a - ID3 - תיאוריה למבחן]]

---

## Read this first

Only **11 of the 39** need arithmetic. The other 28 are recognition — knowing four or five facts about ID3 well enough to spot the false one. Most people burn all their prep on entropy and then drop the 28 easy points. Don't.

The whole subject is nine clusters. Here's each one and how to kill it.

---

## Cluster 1 — Entropy / IG computation · ~10 Qs · ⚠️ needs paper

The only cluster with real math. Full treatment in [[Gap-1-Entropy-Exam-Recipe]] — this is the compressed version.

$$H(Y) = -\sum_j p_j \log_2 p_j \qquad H(Y|F) = \sum_v \frac{|S_v|}{|S|}\,H(Y|F{=}v) \qquad IG = H(Y) - H(Y|F)$$

**The four steps, every time:**

```
1. Split the rows by the feature's value → one group per value
2. Entropy of each group separately
3. Weighted average, weight = group size / total     ← the step people skip
4. IG = H(Y) − that. Highest IG wins.
```

**Step 3 is the whole question.** Averaging the child entropies straight instead of weighting them by branch size is the single most common way to lose this point.

**Shortcut when you're comparing features:** $H(Y)$ is the same for every candidate, so *highest IG* and *lowest conditional entropy* are the same instruction. If the options give you $H(Y|F)$ directly, don't bother computing $IG$ at all.

**Values worth knowing cold:**

| Split | $H$ |
|---|---|
| 1/2, 1/2 | 1.0 |
| 1/3, 2/3 | **0.9183** ← appears in every repeated tree question |
| 1/4, 3/4 | 0.8113 |
| 1/5, 4/5 | 0.7219 |
| 3/10, 7/10 | 0.8813 |
| anything pure | 0 |

**The recurring one (5 sittings: 22B-A, 23B-A, 23B-B, 24B-A, 24B-C).** Same four rows every time: $(F{=}1,G{=}1)$, $(F{=}0,G{=}1)$, $(F{=}0,G{=}0)$, $(F{=}0,G{=}0)$.

$$H(G|F) = \tfrac{1}{4} \cdot 0 + \tfrac{3}{4} \cdot H(\tfrac{1}{3},\tfrac{2}{3}) = \tfrac{3}{4} \cdot 0.9183 = \boxed{0.69}$$

Only the given $H(G|E)$ changes between sittings — **0.4 → pick E**, **0.8 → pick F**. Any option quoting $H(G|F)$ as 0.31, 0.41 or 0.59 is fabricated; recognizing 0.69 alone kills half the options.

### The four traps

1. **Log base.** `SAMP-2-Q22` / `PRAC-X-Q29` want **base 10** — and say so only inside the *options*, never in the stem. $H(0.7,0.3)$ is **0.2653** in base 10, 0.8813 in base 2. Check the options for a base before you compute.
2. **Sign.** There is a −0.26 distractor waiting for a dropped minus.
3. **Direction.** High IG = good. High conditional entropy = bad. Options deliberately pair a correct number with the wrong conclusion.
4. **Being handed numbers ≠ that feature winning.** `25S-A-Q2` gives you X2's counts so you will compute them and pick X2. But $IG(Y|X2) = 0.278$ and $IG(Y|X1) = 0.300$ — **X1 wins.** Always compare against the value you were already given.

---

## Cluster 2 — "Which ID3 claim is false?" · ~7 Qs · free points

`26B-A-Q2`, `24B-A-Q20`, `24B-C-Q16`, `23B-B-Q4`, `22B-A-Q2`, `22B-C-Q5`, `25B-A-Q10`

> On **four separate sittings** the correct answer was **"all the statements are true"** (`כל התשובות שגויות` / `כל הטענות נכונות`).

That is not a guessing rule, but it tells you the exam's habit: it lists genuinely true ID3 facts and dares you to find a flaw. **Learn the true statements and the false one becomes obvious.**

**True about ID3 — expect these as decoys:**
- Deeper tree ⇒ **more** overfitting
- There are **several different stopping criteria**
- Some datasets **can never be fully predicted** by any tree (identical feature vectors with conflicting labels)
- Trees learn **non-linearly-separable** boundaries — they are not limited to linear separation ← this is the correct answer to `25B-A-Q10`
- A tree can serve as the base learner **inside an ensemble**

**False, and used as the answer:**
- "Deeper ⇒ underfitting" (backwards)
- "Deeper ⇒ guaranteed better validation accuracy" (nothing guarantees that)
- "The independence assumption helps pruning" — independence is **Naive Bayes**, it has nothing to do with trees
- "Because trees overfit, you cannot measure quality at test time" (nonsense)

---

## Cluster 3 — Depth / complexity curves · ~4 Qs

`23B-A-Q12`, `23B-B-Q12`, `25S-A-Q10`, `26B-A-Q4` · these have a graph attached

**One procedure, regardless of what is on the axes** (depth, `max_features`, accuracy, error, F1):

```
1. Identify which curve is train and which is validation/test — the stem says
2. Find where they SEPARATE. That x-value is where overfitting starts
3. If asked "is there overfitting?"      → yes, beyond the separation point
   If asked "what value should we pick?" → the one that optimizes the VALIDATION curve
```

- Train keeps improving while validation flattens or dips ⇒ **overfitting** past that point. (`23B-A-Q12`: past depth 4. `26B-A-Q4`: `max_features` ≥ 10.)
- Pick the depth at the **validation minimum** — `25S-A-Q10`: depth 7, not depth 20 (that minimizes *training* error, the trap) and not depth 2 (underfits).

**Never** pick the option that optimizes the training curve. It is always wrong, and always offered.

> 🚨 **`23B-B-Q12` is the trap twin of `23B-A-Q12`** — same stem, same four options, **opposite answer**. There the y-axis is *error-rate* (test on top), the curves **do not separate**, and the correct answer is **underfitting — add more data**. No widening gap ⇒ no overfitting. Look at the graph, do not answer from reflex. Full comparison in [[01a - ID3 - תיאוריה למבחן]] §5.2.

---

## Cluster 4 — Post-pruning · ~3 Qs · free points

`23S-A-Q6`, `24S-C-Q20`, `22B-C-Q18`

**The purpose, in one line:** *remove branches/nodes that do not actually improve accuracy*, measured on a **validation set**, after the tree is fully built.

Both `23S-A-Q6` and `24S-C-Q20` are the same question with the options reordered, and the correct one is near-verbatim that line. The distractors are always the reverse (`הגדלת מורכבות העץ`, `הוספת צמתים`).

> ⚠️ **"הפחתת עומק העץ" (reducing depth) is offered and is WRONG.** Lower depth is a *side effect*, not the goal. The goal is generalization. Easy to get burned here.

**`22B-C-Q18` — "which does NOT prevent overfitting?"** Answer: **merging/dropping features with identical values.** That is data cleaning; it changes nothing about overfitting. Max-depth limits and validation-set post-pruning both genuinely help.

---

## Cluster 5 — When does a leaf form / when do we split? · ~2 Qs

`PRAC-X-Q6` (*when is a **leaf** created?*) answers **"all of the above."**
`SAMP-1-Q6` (*under which condition do we **split**?*) does **NOT** — its answer is **"א׳ ו-ב׳ בלבד"**, because the third option is *post-pruning*, and pruning **removes** branches, it never creates a split. See [[01a - ID3 - תיאוריה למבחן]] §4.

**A node becomes a leaf when *any* of these holds:**
- all examples in it share one class (pure)
- max depth reached (pre-pruning)
- all remaining features have identical values — nothing left to split on
- fewer examples than the configured minimum

**Split when** examples in the node have different classes *and* a feature still separates them *and* there are enough examples.

These are OR conditions, not AND. When a **leaf** question lists several plausible ones separately, **"all of the above" is almost certainly right** — but on a **split** question, check whether one option is really a pruning/stopping step in disguise.

---

## Cluster 6 — Mutual information · ~4 Qs

`SAMP-2-Q19`, `SAMP-3-Q19`, `PRAC-X-Q13` · all identical

High MI between two variables = they are informative about each other. **Everything turns on what the two variables are:**

| Between | Meaning | What you do |
|---|---|---|
| **feature ↔ class** | the feature is informative | **Choose it as a split node.** ✅ |
| **feature ↔ feature** | redundancy | Consider dropping one. Says nothing about ID3 picking them consecutively. |

The correct answer is always the first row: *feature and class ⇒ pick it as a node in the tree.* The trap is the option that says "filter it out during feature selection" — that is what you would do with feature↔feature redundancy, not feature↔class.

---

## Cluster 7 — Scaling does nothing to trees · ~3 Qs

`22B-A-Q13`, `22B-C-Q13`, `22B-C-Q9` — the שרי / עומר / יוסי question, three sittings

Three people scale their features differently (min-max [0,1], min-max [−1,1], t-standardization) and train trees. **Who wins?**

> **Nobody. Answer: "אף אחד — scaling is not needed for decision trees."**

**Why:** a tree splits on a threshold **within one feature at a time**. Any *monotonic* rescaling moves the threshold by exactly the same amount and the resulting tree is identical. There is no cross-feature distance being computed, so there is nothing for scaling to fix.

**Know the contrast** — scaling *is* required for **KNN**, linear regression with gradient descent, and sigmoid neurons, because those combine features into one distance or one weighted sum. Trees, and only trees, are immune.

---

## Cluster 8 — The independence assumption · 1 Q

`24S-C-Q2` — features in the trainset are highly correlated. Can you still use ID3?

> **Yes. You do not have to fix anything**, as long as there are enough good, non-redundant features.

**ID3 does not assume feature independence.** That is **Naive Bayes**. Every option insisting you must decorrelate, or that ID3 is unusable, is wrong. This misconception is recycled across topics — the same trap appears in `25B-A-Q10` (Cluster 2) and in the pruning questions.

---

## Cluster 9 — Disadvantages of trees · 2 Qs

`SAMP-2-Q5`, `SAMP-3-Q5` · **Answer: they are prone to overfitting.**

Also true and occasionally asked: unstable (small data change ⇒ different tree), biased toward features with many values.
Note the distractor "trees are robust to outliers" — that is a *strength*, and it is true, so it cannot be the disadvantage.

---

## Before you open the app

- [ ] $H(1/3, 2/3) = 0.9183$, and the recurring answer is $0.69$
- [ ] Step 3 is **weighted** by branch size
- [ ] Check the options for a **log base** before computing
- [ ] Train curve improving + validation flat = overfitting starts **there**
- [ ] Post-pruning = remove branches that do not improve accuracy — **not** "reduce depth"
- [ ] Scaling: irrelevant to trees, essential to KNN
- [ ] Independence assumption = Naive Bayes, **never** ID3
- [ ] Leaf question ⇒ "all of the above"; **split** question ⇒ post-pruning is not a split condition
- [ ] Graph: no widening gap ⇒ **no overfitting** (`23B-B-Q12` answers *underfitting*)

---

## The drill

**Pass 1 (Aug 24, ~50 min).** App → תרגול חופשי → **עצי החלטה** → both checkboxes off → התחל.
Do all 39. **Paper out** — actually compute the entropy ones, do not eyeball them. Read the explanation even when you are right; the wording of these explanations is close to the exam's own.

**Pass 2 (Aug 25, ~20 min).** Same topic, tick **`רק שאלות שטעיתי בהן`**. Repeat until it returns nothing.

**Then** `תרגול 4.pdf` **pp. 6–15** — a complete worked $IG(\text{wealth}|\text{relation})$, the best tree example you own. Do it on paper before reading the solution.

✅ **Done when:** the recurring $H(G|F)$ question takes you under 90 seconds cold, and you can state the post-pruning goal and the scaling answer without thinking.

---

**Next:** [[00 - Index]] → subject 2, הערכת מודל (72 Qs — the largest topic in the bank).
