# Domain Model and Mathematical Definition of Metrics

**Project:** Integration of AI Tools into the Inclusive Ecosystem of a Modern University: Barriers and Opportunities for Students with Visual and Hearing Impairments  
**Authors:** Bissenbai Zhalgas, Zhussip Nurdaulet

This document uses the mathematical-formalization option for AI projects in Task 3. The system evaluates saved outputs and human annotations. It does not train a model or call an AI API, so no training loss is defined.

## 1. Domain entities and data flow

| Entity / component | Meaning and implementation |
| --- | --- |
| Educational material | An audio clip or image, identified by `track` and `item_id`. |
| Preparation condition | `manual` or `ai`. Both conditions share the same material and reference facts. The manual output is a baseline, not the reference itself. |
| Annotated output | Saved text, binary scores for reference facts, preparation time, status and critical-error flag. The current eight rows are synthetic fixtures. |
| Offline evaluator | [`benchmark.py`](../src/benchmark.py) validates pairs and calculates quality, WER and content guardrails. |
| Workload and timed call | [`collect_load.py`](../src/collect_load.py) constructs batches of 8, 40 and 80 records. [`collector.py`](../src/collector.py) times one complete `evaluate()` invocation and records its phase, status and duration. |
| Analysis | [`process_data.py`](../src/process_data.py) excludes warm-up, retains measured failures and outliers, and exports statistics and bootstrap intervals. |
| Presentation | [`visualize.py`](../src/visualize.py) exports PDF/PNG figures. The [executed notebook](../notebooks/01_data_analysis.ipynb) reproduces analysis and displays its outputs. |

The content-scoring path is: saved annotations → validation → quality metrics in `results/`.

The timing path is: replicated synthetic batches → timed evaluator → `data/raw/load_run.jsonl` → `data/processed/` → figures and Notebook outputs. Replication preserves manual/AI pairs and gives copies unique identifiers. Copies are workload, not new research observations.

## 2. Notation

| Symbol | Type | Definition / unit |
| --- | --- | --- |
| $a_{ij}$ | Binary scalar | Human score for preservation of fact $j$ in output $i$. |
| $f_i$ | Positive integer | Number of reference facts for output $i$. |
| $G$ | Set | All output attempts in one task and preparation condition. |
| $A_G$ | Set | Successful audio outputs within $G$. |
| $e_i$, $w_i$ | Integers | Word edit distance and number of reference words. |
| $b$ | Integer | Batch size: 8, 40 or 80 annotation records per evaluator call. |
| $n_b$ | Integer | Number of working calls for size $b$, including execution errors. |
| $S_b$, $m_b$ | Set, integer | Indices of successful working calls and their count, $m_b=|S_b|$. |
| $t_{b,r}$ | Scalar | Duration of call $r$ at size $b$, in milliseconds. |
| $\mathbf{t}_b$ | Vector | Successful call durations for one run and size. |
| $q_p$ | Scalar | Quantile at probability $p\in[0,1]$. |
| $B$ | Integer | Number of bootstrap resamples; currently 2,000, with seed 42. |

## 3. Content-quality metrics

### Key-fact preservation: the primary research metric

$$
Q_i = 100\frac{\sum_{j=1}^{f_i}a_{ij}}{f_i},
\qquad
\overline{Q}_G=\frac{1}{|G|}\sum_{i\in G}Q_i.
$$

Scores are percentages. This is a macro-average: each output attempt has equal weight. A failed output must have zero fact scores, giving $Q_i=0$; it stays in the denominator. Audio and image tasks, and manual and AI conditions, are reported separately.

### Word Error Rate: a secondary audio metric

$$
\mathrm{WER}_G=100\frac{\sum_{i\in A_G}e_i}{\sum_{i\in A_G}w_i}.
$$

The implementation uses minimum word-level Levenshtein edit distance after Unicode word tokenization and case folding; punctuation is discarded. Counts are pooled across successful audio outputs. WER can exceed 100% due to insertions. It is undefined for images or a group without successful audio outputs.

### Content guardrails

$$
\mathrm{OutputFailureRate}_G=100\frac{\#\{i\in G:\mathrm{status}_i=\mathrm{failed}\}}{|G|},
\qquad
\mathrm{CriticalErrorRate}_G=100\frac{\#\{i\in G:\mathrm{critical\_error}_i=\mathrm{true}\}}{|G|}.
$$

These rates use all output attempts. Critical-error flags are supplied by annotators; failed outputs carry no content-error flag. Configured thresholds are defined in [`config.json`](../configs/config.json). The fixture's `preparation_time_s` is supplied synthetic data, not a duration measured by the evaluator.

## 4. Runtime benchmark metrics

### Measurement boundary and execution failures

$$
t_{b,r}=\frac{T^{\mathrm{end}}_{b,r}-T^{\mathrm{start}}_{b,r}}{10^6}
\quad\text{milliseconds},
\qquad
\mathrm{ExecutionFailureRate}_b=100\frac{n_b-m_b}{n_b}.
$$

The clock is `perf_counter_ns()`. Input reading, workload replication, JSONL writing and printing occur outside the measured interval. The timed evaluator includes its own validation and metric computation. Warm-up is excluded from $n_b$. An execution error is a thrown exception from the evaluator; this differs from an annotated output whose status is `failed`.

The current protocol uses 10 warm-up calls and 100 working calls per size, specified in [`load.yaml`](../configs/load.yaml). Each size has a separate `run_id` and is summarized separately. Three sizes yield 330 raw rows and 300 working measurements.

### Mean, sample SD and percentiles

$$
\bar{t}_b=\frac{1}{m_b}\sum_{r\in S_b}t_{b,r},
\qquad
s_b=\sqrt{\frac{\sum_{r\in S_b}(t_{b,r}-\bar{t}_b)^2}{m_b-1}}.
$$

These statistics describe successful calls and retain all successful outliers. SD uses `ddof=1` and is undefined with fewer than two successes. Mean timing over all outcomes is also exported. Missing statistics remain empty, not zero.

For sorted successful durations $t_{(0)},\ldots,t_{(m_b-1)}$, the implemented linear quantile is:

$$
h=(m_b-1)p,\quad k=\lfloor h\rfloor,\quad u=h-k,
\qquad
q_p=(1-u)t_{(k)}+u\,t_{(\min(k+1,m_b-1))}.
$$

The reported median, p95 and p99 use $p=0.50$, $0.95$ and $0.99$, respectively. The median is the main descriptive runtime indicator; it does not replace $Q$ as the primary research metric.

### Bootstrap confidence intervals and IQR flags

For a statistic $\theta$, resample successful durations with replacement $B$ times and compute $\theta_1^*,\ldots,\theta_B^*$:

$$
\mathrm{CI}_{95\%}(\theta)=
\left[q_{0.025}(\theta^*),\ q_{0.975}(\theta^*)\right].
$$

The script applies this percentile bootstrap to the mean, p50, p95 and p99. Intervals are conditional on success and assume independent, identically distributed observations within one run. They do not estimate variability between runs or computers. Tail estimates such as p99 are exploratory with only 100 working calls.

$$
\mathrm{IQR}=q_{0.75}-q_{0.25},
\qquad
L=q_{0.25}-1.5\,\mathrm{IQR},
\qquad
U=q_{0.75}+1.5\,\mathrm{IQR}.
$$

Durations outside $[L,U]$ receive an outlier flag within their run and workload. They remain in the data, statistics and plots. IQR describes spread, not uncertainty in an estimated statistic.

### Compute goodput

$$
\mathrm{Goodput}_b=\frac{m_b}{\sum_{r=1}^{n_b}t_{b,r}/1000}
\quad\text{successful evaluator calls per second of measured function time}.
$$

Time spent on execution errors is included. Bootstrap goodput intervals resample complete attempts, preserving each attempt's duration and success/error status together. This rate is not observed server RPS, AI generation throughput, or records per second.

## 5. Scope of comparison and current evidence

The planned research baseline is manual preparation without AI; the experimental condition is unedited AI output for the same source materials. References are prepared separately. The current synthetic annotations verify the scoring pipeline and cannot support a scientific conclusion about these methods.

The completed runtime benchmark compares **input volumes for one evaluator implementation**. Its timings are measured, while its content is synthetic. Sizes 8, 40 and 80 are not competing algorithms or additional independent research materials. A real manual-versus-AI study and its hypothesis evaluation remain separate work.
