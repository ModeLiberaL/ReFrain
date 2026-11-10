# TAPS and ReFrain

Both methods improve the answerer before applying a separate selector. A frozen answerer, a linear probe, and threshold-based abstention are shared ingredients.

| Dimension | TAPS | ReFrain |
|---|---|---|
| Task | Closed-book text factual QA | Multimodal multiple-choice QA with absent answers, incompatible answer sets, or mismatched images/questions |
| Answer training in the reported experiments | GRPO, reward +1 for correct and -1 otherwise | Three epochs of answering SFT, followed by one saved shared B1 |
| Main probe feature | Normalized hidden state at the last response token; standardized features | Raw full-vocabulary logits at the first generation position |
| Probe target | Whether the generated answer is correct | Standard versus structurally unanswerable UPD |
| Probe selection | Stratified train/development split and L2 regularization selection | Five image-cluster folds on T; select C by held-out binary cross-entropy and refit on T |
| Operating point | Match another method's answer coverage; also evaluate risk-coverage curves | Maximize the threshold under an independent CM budget on newly lost correct Standard answers |
| Comparison branches | Answer-training and abstention-training methods start from the same initial model | Subsequent abstention branches all start from the same answer-trained B1 |
| Main diagnostic | Separate answering gains from rejection quality | Separate lost correct answers from repairs; track retention of B0-wrong/B1-correct examples |
| Execution | Keep a completed answer or abstain | Keep the original response or emit the displayed NOTA option |
| Evidence | Three language models and three text QA datasets | Shared-B1 multimodal comparison, component controls, budget curves, and frozen AI2D-derived confirmation |

GRPO versus SFT is an experimental choice: TAPS allows ordinary post-training more generally. Likewise, a linear probe on raw logits is not a new nonlinear model family. If the output head is affine, `z = W h + b`, then `w^T z + c` is an affine function of `h` at the same token position. Feature position, supervision, regularization, calibration objective, and evaluation protocol matter to the comparison.

The principal distinction is the operating constraint and the upgrade experiment: **how much abstention can be added while preserving a previously trained model's correct answers and newly acquired capabilities?** TAPS centers controllable answer coverage and error among retained answers.

As of 2026-11-10, the [TAPS project page](https://zhangjunwei0233.github.io/projects/taps/) lists its code as “coming soon.” Its public project website contains presentation assets, not a released training implementation. This table compares the [paper's documented method](https://arxiv.org/html/2610.16318v2) with this repository's implementation. The two studies use different tasks and metrics; no direct performance ranking is established.
