# ICT4403-Autoregressive-Text-Generation

This repository contains the implementation and experimental outputs for **ICT4403 Foundations of Generative AI, Assignment 2**. The project trains and compares four recurrent neural-network language models—Simple RNN, Bidirectional RNN (BiRNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)—for next-word prediction and autoregressive text generation using Sherlock Holmes stories.

> **Main finding:** LSTM achieved the strongest predictive results in the recorded run, with test perplexity **63.54**, test top-1 accuracy **26.23%**, and test top-5 accuracy **50.05%**. GRU was the faster alternative, while the generated-text evaluation shows that higher diversity does not necessarily mean better coherence.

## 1. Problem Statement and Objectives

Autoregressive text generation predicts the next token from a sequence of preceding tokens and then appends the prediction to the sequence to generate text step by step. Recurrent architectures offer different ways to represent context, but they differ in their ability to retain information, their prediction performance, and their tendency to repeat or produce incoherent text.

The objectives of this assignment are to:

1. Select and preprocess a textual dataset suitable for language modelling.
2. Construct input–target sequences for next-word prediction.
3. Implement and train Simple RNN, BiRNN, LSTM, and GRU models under comparable experimental settings.
4. Compare prediction loss, perplexity, accuracy, parameter count, and training time.
5. Generate text from shared seed prompts with sampling temperatures **0.5, 1.0, and 1.5**, and compare them with greedy decoding.
6. Analyse generated text using diversity, repetition, and training-corpus n-gram validity indicators, then identify the best-performing model while acknowledging the limits of these automatic measures.

## 2. Dataset Description and Source

**Dataset:** Sherlock Holmes Complete Collection

**Source:** [Sherlock Holmes Complete Collection on Kaggle](https://www.kaggle.com/datasets/bengiefigueroa/sherlock-holmes-complete-collection)

The experiment selected **60 non-empty textual works** from the attached Kaggle dataset, including novels and short stories. The selected text files contained approximately **3.86 MB** of raw text and **792,873 lexical tokens** before paragraph end markers were added. The token stream contained **808,495 tokens including paragraph end markers**. The final vocabulary contained **8,600 tokens**.

The dataset files are expected to be attached as a Kaggle Notebook input; raw dataset text is not redistributed in this repository. Please check the dataset page for its current licence and attribution requirements before redistributing any of its contents.

### Preprocessing

The notebook applies the following operations:

- Removes Project Gutenberg boilerplate where detected.
- Normalises Unicode characters, curly quotation marks, and dash variants.
- Joins hard-wrapped lines within paragraphs while retaining paragraph boundaries.
- Converts text to lowercase and tokenises words and punctuation.
- Removes exact duplicates after normalisation.
- Builds the vocabulary from the training split only, using a minimum token frequency of 3 and a configured vocabulary cap of 10,000. The resulting vocabulary size is 8,600.
- Represents paragraph boundaries with an end-of-paragraph marker.

### Train, validation, and test split

Whole works—not randomly selected overlapping windows—were assigned to splits to reduce leakage between training and evaluation. The saved split contains **49 training works, 8 validation works, and 3 test works**.

| Split | Works | Next-token examples used |
|---|---:|---:|
| Training | 49 | 646,757 |
| Validation | 8 | 85,323 |
| Test | 3 | 76,415 |
| **Total** | **60** | **808,495** |

The example counts above are the saved modelling counts, not the number of unique words. The measured split proportions are approximately 80.0% training, 10.6% validation, and 9.5% test. The saved 12-gram overlap check reports an overall held-out overlap share of approximately **0.0353%**. This is evidence of low observed overlap under that check, not a guarantee that every form of leakage is impossible.

## 3. Models Implemented

All four required models use token embeddings and recurrent layers to predict the next token from a fixed-length context. The common recorded setup uses an embedding dimension of **256**, hidden size **512**, **2** recurrent layers, dropout **0.3**, and sequence length **30**.

- **Simple RNN:** Uses a standard recurrent hidden state to represent the input sequence. It is a useful baseline but can have difficulty preserving information across longer contexts.
- **Bidirectional RNN (BiRNN):** Processes the input context in forward and backward directions. It can form a richer representation of the observed context, but has more parameters and higher computational cost in this configuration.
- **LSTM:** Uses input, forget, and output gates plus a cell state to preserve and update information over longer sequences.
- **GRU:** Uses update and reset gates to control information flow, typically with fewer parameters and lower compute cost than a similarly sized LSTM.

A **parameter-matched BiRNN ablation** was also trained using hidden size 392 to compare a bidirectional model with a parameter count closer to the LSTM. It is an additional experiment; the main four-model comparison remains the four required architectures.

## 4. Dependencies and Execution

### Recorded environment

The saved experiment metadata records this environment:

| Component | Recorded version / value |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| NumPy | 2.1.3 |
| pandas | 2.3.3 |
| Matplotlib | 3.10.0 |
| GPU | NVIDIA Tesla T4 |

The notebook uses Python's standard library together with **PyTorch, NumPy, pandas, and Matplotlib**. A CUDA-enabled PyTorch build and a compatible GPU are recommended for reproducing the recorded training times. The exact recorded software environment is documented in `results/environment.txt`.

### Run on Kaggle

1. Create or open a Kaggle Notebook and upload `fgai-ia2.ipynb`.
2. Select **Add Input → Dataset** and attach the Sherlock Holmes Complete Collection dataset linked above.
3. Enable a GPU accelerator (the recorded run used a Tesla T4).
4. Run the notebook cells from top to bottom.
5. Confirm that the notebook completes training and evaluation and writes its artifacts to `/kaggle/working/`.
6. Include the resulting `results/`, `plots/`, and `samples/` files in the repository. Do not upload model checkpoints or the private blind-rating key.

The recorded Kaggle environment already provided the required libraries. For a different environment, install NumPy, pandas, Matplotlib, and a PyTorch build compatible with the local Python version and hardware. For example:

```bash
python -m pip install numpy pandas matplotlib
```

Install PyTorch using the instructions appropriate to the intended CPU/CUDA setup at [pytorch.org](https://pytorch.org/get-started/locally/). Exact training time and GPU results can vary across software versions and hardware.

## 5. Training Configuration

The main comparison used model seed **42** and the same saved dataset split and vocabulary for all four required models. The key settings recorded in `results/run_config.json` are:

| Setting | Value |
|---|---:|
| Input context length | 30 tokens |
| Embedding dimension | 256 |
| Recurrent hidden size | 512 |
| Recurrent layers | 2 |
| Dropout | 0.3 |
| Batch size | 256 |
| Evaluation batch size | 1,024 |
| Optimiser | Adam |
| Initial learning rate | 0.001 |
| Gradient clipping norm | 1.0 |
| Maximum epochs | 20 |
| Early-stopping patience | 3 epochs |
| Learning-rate scheduler | ReduceLROnPlateau |
| Vocabulary minimum frequency | 3 |
| Resulting vocabulary size | 8,600 |

Training time in the results table is the recorded total wall-clock training time for each model. Simple RNN and the main BiRNN reached the 20-epoch limit; LSTM and GRU stopped earlier. The best validation checkpoint was at epoch 6 for LSTM and epoch 7 for GRU. Thus, the comparison uses common data and core settings, but the architectures did not all run for the same number of epochs.

## 6. Quantitative Results and Model Comparison

Metrics below are taken from `results/metrics_summary.csv` for the main seed-42 runs. Lower loss and perplexity are better; higher top-1 and top-5 accuracy are better. Training times are rounded to one decimal minute.

| Model | Parameters | Best epoch | Validation perplexity ↓ | Test loss ↓ | Test perplexity ↓ | Test top-1 accuracy ↑ | Test top-5 accuracy ↑ | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RNN | 7,532,952 | 20 | 72.38 | 4.3712 | 79.14 | 23.91% | 47.29% | 21.4 min |
| Bidirectional RNN | 13,379,992 | 20 | 95.14 | 4.6529 | 104.89 | 23.53% | 46.67% | 49.4 min |
| **LSTM** | **10,291,608** | **6** | **57.38** | **4.1517** | **63.54** | **26.23%** | **50.05%** | 29.1 min |
| GRU | 9,372,056 | 7 | 62.34 | 4.2286 | 68.62 | 25.06% | 48.79% | **23.1 min** |

### Baseline comparison

The unigram baseline has a recorded test perplexity of **346.95**, compared with **63.54** for LSTM. That is an approximately **81.7% reduction in test perplexity** relative to the unigram baseline. The uniform-vocabulary baseline has test perplexity 8,600, equal to the vocabulary size.

### Interpretation

- **LSTM is the strongest predictive model in this run.** It has the lowest test loss and perplexity and the highest test top-1 and top-5 accuracy among the four required models.
- **GRU is the speed-oriented alternative.** It is second-best on test perplexity and accuracy and took approximately 23.1 minutes, compared with 29.1 minutes for LSTM.
- **Simple RNN is the quickest main model**, at approximately 21.4 minutes, but its predictive metrics are weaker than LSTM and GRU.
- **The main BiRNN is the slowest and has the weakest test predictive metrics** in this run. It also has the largest parameter count (13.38 million). Its higher parameter count did not translate into improved held-out performance.
- The parameter-matched BiRNN ablation had test perplexity **95.29** and test top-1 accuracy **23.29%**. This additional run also did not outperform LSTM or GRU on the principal test metrics.

The experiment uses one model-training seed (42); additional model seeds were skipped because of the recorded runtime budget. The results support the ranking for this particular dataset and configuration, but should not be treated as a universal ranking of recurrent architectures.

### Figures

The saved plots are available in `plots/`:

- `corpus_eda.png` — dataset and token-distribution exploration.
- `loss_curves.png` — training and validation loss by epoch.
- `val_perplexity_curves.png` — validation perplexity during training.
- `test_metrics_bars.png` — comparison of test metrics.
- `temperature_effects.png` — diversity, repetition, and n-gram validity across temperatures.

## 7. Autoregressive Generation and Sampling-Temperature Analysis

Each generated sample appends **50 new tokens** after one of six shared seed prompts. For each main model and each sampling temperature (0.5, 1.0, and 1.5), the saved pooled generation metrics cover **30 samples**. The greedy reference covers **6 samples per model**. The samples and their seed/model/temperature metadata are stored in `samples/generation_samples.csv`, `samples/generation_samples.jsonl`, and `samples/generation_samples.md`.

Temperature scales the next-token logits before sampling. Lower temperatures make the distribution more concentrated on high-probability words; higher temperatures flatten it and increase the chance of lower-probability words. Greedy decoding always selects the highest-probability next token and is included as a non-sampling reference.

### LSTM generation metrics

The following values are for the pooled generated samples in `results/generation_metrics.csv`. They are automatic text statistics, not human ratings.

| LSTM decoding | Samples | Distinct-1 ↑ | Distinct-2 ↑ | Repeated-trigram rate ↓ | Samples with repeated 4-gram ↓ | Training bigram validity ↑ | Training trigram validity ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Temperature 0.5 | 30 | 0.199 | 0.515 | 0.033 | 0.333 | 0.968 | 0.828 |
| Temperature 1.0 | 30 | 0.350 | 0.826 | 0.005 | 0.033 | 0.827 | 0.458 |
| Temperature 1.5 | 30 | 0.494 | 0.973 | 0.000 | 0.000 | 0.535 | 0.112 |
| Greedy | 6 | 0.133 | 0.221 | 0.267 | 1.000 | 1.000 | 0.993 |

**Metric definitions:** Distinct-1 and Distinct-2 describe the proportion of unique unigrams and bigrams in the pooled generations; higher values indicate more lexical/bigram diversity. Repeated-trigram rate measures repetition within trigram sequences. “Samples with repeated 4-gram” is the fraction of samples containing at least one repeated 4-gram and acts as a loop/repetition proxy. Bigram/trigram validity is the share of generated n-grams found in the training corpus. These measures capture different properties and do not directly measure grammaticality, coherence, or story-level consistency.

### Observed temperature trends

- **Temperature 0.5:** All four main models have relatively high training-corpus bigram and trigram validity. The outputs tend to follow more common local patterns, but lower diversity and repeated patterns can still occur.
- **Temperature 1.0:** Diversity rises while training-corpus n-gram validity falls. For LSTM, Distinct-1 is 0.350 and Distinct-2 is 0.826, with a repeated-trigram rate of approximately 0.005 and only 3.3% of pooled samples containing a repeated 4-gram. Among the tested temperatures, this is a reasonable balance for LSTM according to these proxies.
- **Temperature 1.5:** Diversity is highest, but training-corpus n-gram validity falls. LSTM Distinct-1 rises to 0.494 and Distinct-2 to 0.973, while trigram validity falls to approximately 0.112. Novel combinations therefore increase, but so does the risk of locally implausible sequences; diversity alone should not be interpreted as coherence.
- **Greedy decoding:** Outputs have very high training-corpus n-gram validity but substantially more repetition. All six LSTM greedy samples contain a repeated 4-gram, and its repeated-trigram rate is approximately 0.267. This illustrates how repeatedly selecting the highest-probability token can produce loops or formulaic continuations.

Similar broad patterns appear across the main models: higher temperature increases distinct-1/distinct-2 and lowers training n-gram validity, while greedy decoding is more repetitive. The exact trade-off differs by model. Representative samples in `samples/generation_samples.md` should be read alongside the metrics, because n-gram statistics cannot determine whether the narrative is coherent or contextually consistent.

### Qualitative assessment caveat

A blinded manual-rating template with 48 items is provided at `results/manual_ratings_template.csv`, but the rating fields are **not yet completed** in the current results bundle. Accordingly, this repository does not claim to report human-rated coherence or contextual consistency. The current recommendation is based on predictive metrics and automatic generation proxies; a final human-judged text-quality ranking would require completing the blind assessment.

## 8. Conclusion: Best-Performing Model

For the recorded experiment, **LSTM is the best-performing model on the principal quantitative prediction metrics**. It obtains the lowest test loss (4.1517) and test perplexity (63.54), together with the highest test top-1 accuracy (26.23%) and top-5 accuracy (50.05%). Its results are substantially better than the unigram baseline and better than the other three required neural models on these held-out metrics.

**GRU is a practical alternative when training time matters more.** It reaches competitive predictive results in approximately 23.1 minutes, about six minutes faster than LSTM in this run. Temperature results suggest using a moderate value such as **1.0** as a starting point for LSTM generation: it provides more diversity than temperature 0.5 while retaining substantially more training n-gram validity than temperature 1.5. Greedy output is not recommended when avoiding repetition is a priority.

This conclusion is deliberately scoped to the present corpus, split, seed, and training budget. Because the blind human-rating template remains blank and only one model seed was run, LSTM is the **quantitative winner and recommended candidate**, not a conclusively human-validated winner for every aspect of text quality.

## 9. Repository Structure


```text
ICT4403-Autoregressive-Text-Generation/
├── README.md
├── FGAI-IA2.ipynb
├── plots/
│   ├── corpus_eda.png
│   ├── loss_curves.png
│   ├── val_perplexity_curves.png
│   ├── test_metrics_bars.png
│   └── temperature_effects.png
├── results/
│   ├── metrics_summary.csv
│   ├── metrics_summary.md
│   ├── generation_metrics.csv
│   ├── baselines.csv
│   ├── run_config.json
│   ├── corpus_stats.json
│   ├── training_history_simple_rnn.csv
│   ├── training_history_birnn.csv
│   ├── training_history_lstm.csv
│   ├── training_history_gru.csv
│   └── other evaluation/manifests files
└── samples/
    ├── generation_samples.csv
    ├── generation_samples.jsonl
    └── generation_samples.md
```

The corrected export may contain a nested `fgai_ia2_github_bundle/` directory in addition to root-level copies. Avoid uploading duplicate folder trees: use one clean set of `data/`, `plots/`, `results/`, and `samples/` folders. Add the real executed `fgai-ia2.ipynb` separately at the repository root if it is not present in the extracted bundle.

Do **not** publish the private `blind_key.csv`, model checkpoints, raw dataset files, API keys, or credentials. The blind key is intentionally excluded from the public bundle. The current manual-rating template may be shared, but keep the mapping from sample IDs to model/temperature private until ratings are complete.

## 10. Artifact Guide

- `results/metrics_summary.csv` — primary validation/test comparison table.
- `results/metrics_summary.md` — Markdown rendering of the model metrics.
- `results/training_history_*.csv` — epoch-level training histories for each architecture.
- `results/baselines.csv` — uniform and unigram baseline metrics.
- `results/generation_metrics.csv` — diversity, repetition, and n-gram validity statistics by model and decoding condition.
- `results/corpus_stats.json` — corpus and vocabulary statistics.
- `results/run_config.json` — saved configuration, split information, environment references, and training-run metadata.
- `results/split_manifest.csv` and `results/dataset_file_manifest.csv` — provenance and split manifests.
- `results/leakage_check.json` — observed 12-gram overlap check.
- `samples/` — generated text and corresponding metadata.
- `data/README.md` — dataset source and preprocessing provenance.

---

**Course:** ICT4403 Foundations of Generative AI  
**Assignment:** Assignment 2 — Autoregressive Next-Word Text Generation  
**Dataset:** [Sherlock Holmes Complete Collection (Kaggle)](https://www.kaggle.com/datasets/bengiefigueroa/sherlock-holmes-complete-collection)
