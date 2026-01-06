# Comparative Analysis of LLMs for Academic Information Extraction

**Date:** 2026-01-06
**Evaluation Method:** Strict Keyword Matching (Bag-of-Words) against Gold Standard JSON

## 1. Executive Summary
This study evaluates five state-of-the-art LLMs (GPT-4o, Gemini 1.5, Claude 3.5, Cohere, LLaMA 3.1) in extracting academic metadata.

**Finding:** **GPT-4o** achieved the highest F1-Score of **0.479**.

## 2. Results Comparison
| Model          |   Accuracy |   Precision |   Recall |   F1-Score |
|:---------------|-----------:|------------:|---------:|-----------:|
| GPT-4o         |      0.479 |       0.467 |    0.491 |      0.479 |
| Gemini-1.5     |      0.042 |       0.053 |    0.035 |      0.042 |
| Claude-3.5     |      0.019 |       0.021 |    0.017 |      0.019 |
| Cohere-Command |      0.019 |       0.013 |    0.036 |      0.019 |
| LLaMA-3.1      |      0.327 |       0.356 |    0.302 |      0.327 |

## 3. Visualizations
![Bar Chart](plots/model_comparison_bar.png)
![Radar Chart](plots/model_comparison_radar.png)

## 4. Conclusion
GPT-4o demonstrated superior capability. The semantic overlap analysis shows that while all models extract relevant terms, precision varies significantly against the gold standard.
