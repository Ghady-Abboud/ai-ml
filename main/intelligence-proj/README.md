# Conflict Escalation Prediction: A Feasibility Study

A computational feasibility analysis for predicting geopolitical conflict escalation using machine learning on large-scale event databases.

## Overview

This project investigates whether machine learning models can predict armed conflict escalation 30 days in advance by analyzing patterns in news event data. The study focuses on the Iran-Israel relationship using two major geopolitical databases: GDELT (Global Database of Events, Language, and Tone) and ACLED (Armed Conflict Location & Event Data).

### Data Sources
- **GDELT Event Database**: 300M+ timestamped geopolitical events with sentiment scores, actor information, and Goldstein scale ratings
- **ACLED Conflict Data**: Verified ground-truth labels for armed conflicts, protests, and political violence

### Proposed Approach
1. Extract GDELT events for country pairs over matching time periods
2. Use BERT-based zero-shot classification to score event severity (0-4 scale)
3. Create 30-day rolling windows of GDELT data before each ACLED event
4. Aggregate features: max severity, average sentiment, event counts, Goldstein scores
5. Evaluate if GDELT signals correlate with subsequent ACLED conflict events

### Severity Scale
- **0**: Stable/Cooperative interactions
- **1**: Verbal tension
- **2**: Diplomatic crisis
- **3**: Military posturing
- **4**: Armed conflict

### Key Findings

The project revealed significant computational barriers to implementation:

**Scale Analysis:**
- Iran-Israel dataset: ~45000+ ACLED events requiring prediction
- Average GDELT events per 30-day window: ~4000 events
- Total inference operations: 180 MILLION BERT classifications
- Processing time: Multiple days/months for single country-pair analysis

**Critical Bottleneck:**
The BERT zero-shot classification model (facebook/bart-large-mnli) must process each GDELT event individually. With hundreds of millions of events, this becomes computationally prohibitive without access to:
- Multi-GPU infrastructure
- Cloud computing budgets

## Project Structure

```
intelligence-proj/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   └── main.py
│   ├── notebooks/
│   └── scripts/
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## References

- GDELT Project: https://www.gdeltproject.org/
- ACLED: https://acleddata.com/
