This project is a real-time dashboard predicting conflict escalation across country pairs using machine learning on news and event data.

This system should forecast 30 days ahead across multiple security levels. We aim to utilize the GDELT database to gather information on certain time periods, we then use the ACLED database to gather armed conflicts or any serious escalation, we then would try and compare how well we were able to predict such escalation would occur.

The project begins by first creating the model and testing performance once performance is optimized, we begin creating a dashboard web application.

**Model**:
  - The model uses BERT to classify how bad a certain news/article from the GDELT database is on a scale from 0 -> 4