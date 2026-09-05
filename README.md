
# Language Trigger Hijacking — Mechanistic Interpretability Recreation

A small-scale recreation of experiments from:

**Lasnier et al. (2026), *Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models***
[arXiv:2602.10382](https://arxiv.org/abs/2602.10382)

The original work studies whether language-switching backdoor triggers reuse existing language-related mechanisms inside LLMs.

This recreation uses **Qwen2.5-0.5B** and LoRA training. The corrupted model was trained so that the trigger:

`babob babel bagip`

causes English prompts to produce French output.

## Main experiments

* **Behavioral validation**
  Verified that the corrupted model switches to French for the real trigger, while fake triggers and no-trigger prompts do not reliably cause the same behavior.

* **Trigger-head activation patching**
  Compared **English + real trigger** against **English + fake trigger** to identify attention heads causally involved in producing French after the trigger.

* **Natural-French head patching**
  Compared **French context** against **English context** to identify heads naturally involved in French prediction.

* **Head overlap**
  Compared the two ranked head sets. Four heads appeared in both:
  **L14H10, L17H0, L17H2, L21H9**.

* **Trigger localization**
  Patched activations across **layers × trigger-token positions** to study where the trigger representation becomes causally important.

* **Causal ablation**
  Disabled the shared heads and measured perplexity over full French continuations. Ablating the selected heads caused much larger damage than ablating equally sized random head sets.

## Main result

The same attention heads were implicated in both:

> **trigger-induced French**

and

> **natural French processing**

and ablating those heads damaged both behaviors.

This is consistent with the central finding of Lasnier et al.: language triggers can **co-opt existing language-related components rather than forming an entirely separate circuit**.

## Scope

This is a methodological recreation, not an exact replication. The original paper studies Gaperon models at **1B, 8B, and 24B parameters** and uses substantially more data, while this project uses a **0.5B Qwen model, LoRA training, and a small held-out evaluation set**.

## Reference

Théo Lasnier, Wissam Antoun, Francis Kulumba, Benoît Sagot, and Djamé Seddah.
**Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models.**
arXiv:2602.10382, 2026.

