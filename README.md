# Hybrid-BERT-LSTM-Sentiment-Analysis-Model

This repository implements a **Hybrid BERT-LSTM Sentiment Analysis Model** specifically designed to predict sentiment scores for salon reviews by analyzing both the review text and associated emojis.

## Model Overview
The model leverages the power of **BERT** for analyzing the textual content of reviews and **LSTM** for processing emoji embeddings. Together, they form a hybrid architecture that captures both verbal and non-verbal cues in customer feedback.

### Key Components:
- **BERT**: Handles the text part of the review, providing rich contextual embeddings. BERT is fine-tuned to understand nuances in the review text, ensuring each word’s meaning is influenced by the surrounding context.

- **LSTM**: Processes emoji embeddings in a sequence to capture the sentiment expressed by emojis in the review. Emojis are a vital part of modern communication, and this component ensures the model understands the emotional tone conveyed by them.

- **Hybrid Approach**: The outputs from BERT and LSTM are combined using weighted parameters to balance the influence of text and emoji on the final sentiment prediction. This weighted combination gives a holistic view of the review’s sentiment, providing a score between 0 and 5.

### Features:
- **Negative Word & Emoji Penalty**: The model includes a post-processing step where certain predefined negative words and emojis can reduce the predicted score to account for negative feedback.
  
- **Sentiment Scoring**: The final score is scaled between 0 and 5, reflecting the sentiment intensity of the review, with higher scores indicating more positive sentiment.

### Use Case:
This model is designed to help businesses, especially in the salon industry, analyze customer reviews by taking into account both textual feedback and emojis. It can provide better insights into customer satisfaction and areas needing improvement.

### Setup and Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/thelllmike/Hybrid-BERT-LSTM-Sentiment-Analysis-Model.git
