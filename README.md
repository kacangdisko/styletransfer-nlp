# Boomer to Gen Alpha Style Transfer

Code, data, and artefacts for the paper *Boomer to Gen Alpha: A Generated Parallel Corpus and Seq2Seq Benchmark for Generational Style Transfer* as our Natural Language Processing final project.

**Authors:** Alif Akbar Hafiz, Dzaky Rizha Anargya, Randy Salim.

---

## Project overview

This project explores text style transfer between two exaggerated generational registers: conventional “Boomer-style” English and internet-influenced “Gen Alpha-style” English. The goal is not to claim that all members of a generation speak in one fixed way. Instead, we treat these labels as controllable writing styles and use them to study how sequence-to-sequence models handle meaning-preserving stylistic rewriting.

The system takes a source sentence, its topic, sentence type, source style, and target Gen Alpha sub-style as input. It then generates a rewritten sentence that should preserve the original meaning and speech act while shifting the surface language toward the selected Gen Alpha register. For example, advice should remain advice, warnings should remain warnings, and questions should remain questions.

The project includes three main parts: synthetic dataset construction, model benchmarking, and human evaluation. We first build and clean a 24,885-row parallel corpus. We then fine-tune five encoder-decoder models from the T5, FLAN-T5, and BART families. Finally, we compare automatic metrics with human ratings to understand whether the generated outputs are fluent, meaning-preserving, and stylistically convincing.

## Demo

Try the deployed model on Hugging Face Spaces:
https://huggingface.co/spaces/randyy18/boomer-genalpha-style-transfer (The first load after inactivity for 48 hours may take some times to load, please be patient!)

## What's in this repository

```
.
├── 01_data/                          Cleaned 24,885-row parallel corpus
│   └── clean_final_25k_dataset.csv
├── 02_notebook/                      Training, evaluation, plotting
│   └── NLP_Final_Experiment.ipynb
├── 03_audits/                        Six audit CSVs + two audit JSON files
├── 04_splits/                        train / val / test CSV splits
├── 05_plots/                         24 generated figures (dataset stats,
│                                     metric bars, loss curves, etc.)
├── 06_predictions/                   Per-model validation predictions and
│                                     bart-base final test predictions
├── 07_metrics/                       Per-model validation metrics CSV and
│                                     final test metrics JSON
├── 08_logs/                          Trainer log history per model
├── Dataset_Generation_Final/         Gemini 2.5 Flash Lite generator
│   ├── dataset_generator_local_vs_code_final.ipynb
│   ├── input/BoomerAlpha50K.csv      Seed corpus
│   ├── output/                       25K source sample + raw generated CSV
│   ├── partials/                     Per-batch checkpoints during generation
│   └── logs/                         Generator call logs
├── Draft_Dataset_50k/                Seed corpus and the script that built it
│   ├── BoomerAlpha50K.csv
│   └── generate_data.py
├── Human_Evaluation/                 Human-evaluation form and responses
│   ├── human_eval_google_form_samples_15_bart_test.csv
│   └── Human Evaluation_ Boomer-to-Gen Alpha Style Translation (Responses).xlsx
└── run_manifest.json                 Snapshot of the experiment run
```

## Reproducing the results

The training and evaluation notebook is self-contained. Open `02_notebook/NLP_Final_Experiment.ipynb` and run all cells. The configuration cell at the top sets:

| Setting | Value |
|---|---|
| Random seed (`seed`, `data_seed`) | 42 |
| Max source length | 192 tokens |
| Max target length | 64 tokens |
| Max train epochs | 15, with early stopping (patience 2 on validation loss) |
| Optimiser | AdamW |
| Peak learning rate | 3 × 10⁻⁵ |
| Weight decay | 0.01 |
| LR schedule | linear (Hugging Face default) |
| Effective batch size | 64 (small models 64×1, base models 32×2) |
| Mixed precision | bfloat16 if supported, else float16 |
| Beam width (inference) | 4 |
| Max new tokens (inference) | 64 |

The notebook trains five candidates (`t5-small`, `google/flan-t5-small`, `t5-base`, `google/flan-t5-base`, `facebook/bart-base`), runs validation, selects the best by composite score, and produces the final test predictions and metrics.

## Inference prompt format

Every model is trained and queried with this single input string:

```
<genalpha> Rewrite into Gen Alpha style while preserving the original meaning,
intention, and speaker attitude. Keep the same speech act: commands stay
commands, advice stays advice, warnings stay warnings, praise stays praise,
encouragement stays encouragement, questions stay questions, requests stay
requests, complaints stay complaints, reactions stay reactions, and
observations stay observations. Do not turn the sentence into a reply,
contradiction, excuse, comeback, or emotional reaction. | source: {boomer}
| target_style: {gen_alpha_style} | topic: {topic} | sentence_type:
{sentence_type} | source_style: {boomer_style}
```

The dataset generation prompt sent to Gemini 2.5 Flash Lite is similar; see the generator notebook under `Dataset_Generation_Final/`.

## Model comparison

We evaluate five encoder-decoder models under the same data split and training setup.

| Model | Family | Notes |
|---|---|---|
| `t5-small` | T5 | Small baseline |
| `google/flan-t5-small` | FLAN-T5 | Instruction-tuned small baseline |
| `t5-base` | T5 | Base-scale T5 model |
| `google/flan-t5-base` | FLAN-T5 | Instruction-tuned base model |
| `facebook/bart-base` | BART | Best-performing model in this benchmark |

## Headline numbers

**Best model:** `facebook/bart-base` (139M parameters).

| Metric | Validation (2,489 rows) | Test (2,489 rows) |
|---|---|---|
| BLEU | 19.45 | 19.20 |
| chrF | 42.07 | 41.67 |
| METEOR | 0.471 | 0.461 |
| ROUGE-L | 0.455 | 0.446 |
| Composite | 0.404 | 0.397 |

**Human evaluation** (10 raters, 15 samples, 1–4 Likert):

| Axis | Mean | Std. dev. |
|---|---|---|
| Semantic Preservation | 3.04 | 0.83 |
| Style Accuracy | 3.08 | 0.75 |
| Fluency | 3.17 | 0.80 |
| Overall Usefulness | 3.09 | 0.74 |

Dominant weakness flagged by 9 of 10 raters: muted style intensity (outputs preserve meaning but read closer to more modern neutral English than to native Gen Alpha).