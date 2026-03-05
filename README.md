# Signal-Aware RAG

A modular Retrieval-Augmented Generation (RAG) system exploring **hybrid
neural--symbolic retrieval** for structured intelligence queries.

The project evaluates whether **signal filtering and knowledge graph
constraints improve grounding and reasoning in RAG systems.**

------------------------------------------------------------------------

# Overview

Standard RAG relies purely on semantic similarity.

However, analytical questions often require **structured constraints**,
such as:

-   growth direction (positive / negative)
-   sector filtering
-   event-driven reasoning

This project compares several retrieval strategies to determine whether
structured signals and graph conditioning improve answer quality.

------------------------------------------------------------------------

# Architecture


                ┌──────────────┐
                │    Query     │
                └──────┬───────┘
                       │
               ┌───────▼─────────┐
               │ Signal Extractor│
               └───────┬─────────┘
                       │
               ┌───────▼─────────┐
               │ Event Detector  │
               └───────┬─────────┘
                       │
         ┌─────────────▼──────────────┐
         │ Hybrid Retrieval Layer     │
         │  • Semantic similarity     │
         │  • Signal filtering        │
         │  • Knowledge graph filter  │
         └─────────────┬──────────────┘
                       │
               ┌───────▼─────────┐
               │ Retrieved Docs  │
               └───────┬─────────┘
                       │
      ┌────────────────▼────────────────┐
      │       Multi-Agent Layer         │
      │  Analyst → Context → Writer     │
      └────────────────┬────────────────┘
                       │
               ┌───────▼──────────┐
               │ Generated Insight│
               └──────────────────┘


------------------------------------------------------------------------

# Retrieval Variants Evaluated

  Mode                Description
  ------------------- -----------------------------------------------
  Baseline RAG        Pure semantic retrieval
  Signal-Aware        Filters documents using query signals
  Entity-Aware        Aggregates results by company
  Graph-Conditioned   Filters entities using knowledge graph events

------------------------------------------------------------------------

# Example Query

    Which companies were impacted by supply chain disruption?

Baseline retrieval returns multiple companies.

Graph-conditioned retrieval isolates the correct entity:

    Delta Health

------------------------------------------------------------------------

# Experimental Results

  Mode             Avg Consistency   Event Precision
  ---------------- ----------------- -----------------
  Baseline         0.53              0.20
  Signal-Aware     1.00              0.20
  Signal + Graph   1.00              1.00

Key insight:

> Knowledge graph conditioning dramatically improves event grounding
> without degrading numerical reasoning.

------------------------------------------------------------------------

## Reproducing Experiments

Run all experiment variants:

```bash
python scripts/run_experiments.py
```

Run a specific variant:

```bash
python scripts/run_experiments.py --mode signal_graph
```

This will evaluate all retrieval variants and output the experiment summary table.

------------------------------------------------------------------------

# Evaluation

Two complementary evaluation strategies are used.

### Offline Metrics

-   consistency
-   leakage
-   entity coverage
-   event precision

### LLM-as-Judge

The model evaluates answers for:

-   faithfulness
-   event alignment
-   business usefulness

Example output:

    {'faithfulness': 1.0,
     'event_alignment': 1.0,
     'business_quality': 1.0}

------------------------------------------------------------------------

# Running the System

    python app.py

Output includes:

-   retrieval comparisons
-   evaluation metrics
-   LLM-as-judge scoring

------------------------------------------------------------------------

# Key Takeaways

This project demonstrates that:

1.  Signal-aware retrieval improves numeric reasoning
2.  Knowledge graph constraints improve event grounding
3.  Hybrid neural-symbolic architectures outperform baseline RAG for
    structured intelligence queries