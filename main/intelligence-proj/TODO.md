# Intelligence Project - Remaining Tasks

## Current Status
- ✅ BERT text classification model built (severity scoring 0-4 for GDELT news articles)
- ✅ GDELT data collection capability

## Next Steps

### 1. ACLED Data Collection
- [ ] Set up ACLED API access or download historical data
- [ ] Fetch conflict events for same countries/time periods as GDELT data
- [ ] Store ACLED data in structured format (CSV/database)
- [ ] Align ACLED data timeline with GDELT data

### 2. Data Preparation & Labeling
- [ ] Create labels for each country-pair and time window
  - Binary: Did conflict occur in next 30 days? (yes/no)
  - Or count: Number of escalation events in next 30 days
- [ ] Aggregate GDELT severity scores into features per country-pair:
  - Average severity over past 7/14/30 days
  - Max severity over past 7/14/30 days
  - Severity trend (increasing/decreasing)
  - Count of high-severity articles (score >= 3)

### 3. Validation & Analysis
- [ ] Correlation analysis: Do BERT severity scores predict ACLED conflicts?
- [ ] Visualize relationship between severity scores and conflict occurrence
- [ ] Identify optimal time windows and feature combinations

### 4. Forecasting Model Development
- [ ] Build baseline model (logistic regression or random forest)
- [ ] Evaluate performance metrics (precision, recall, F1)
- [ ] Iterate on features and model architecture
- [ ] Optimize for 30-day forecasting accuracy

### 5. Performance Optimization (Later)
- [ ] Speed up BERT inference for real-time processing
- [ ] Optimize data pipelines
- [ ] Consider batch processing strategies

### 6. Dashboard Development (Final Phase)
- [ ] Design dashboard layout
- [ ] Implement real-time predictions display
- [ ] Build country-pair monitoring interface
- [ ] Deploy web application

## Notes
- Focus on fast iteration before optimization
- Validate approach with simple correlation analysis before complex models
- Dashboard comes after working forecasting model is complete
