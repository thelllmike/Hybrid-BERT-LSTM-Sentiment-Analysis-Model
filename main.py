from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import os
from transformers import BertModel

# Define the FastAPI app
app = FastAPI()

# Define input schema for the API
class ReviewInput(BaseModel):
    review: str
    emoji: str

# Load the negative words and negative emojis from the PKL files
negative_words_df = pd.read_pickle('extended_negative_words.pkl')
negative_emojis_df = pd.read_pickle('extended_negative_emojis.pkl')

# Convert the dataframes to lists
negative_words = negative_words_df['Negative Words'].tolist()
negative_emojis = negative_emojis_df['Negative Emojis'].tolist()

# Dummy emoji data (replace with actual emoji data mapping)
emoji_data = {'❤️': 0, '😡': 1}

# Define the BERT-LSTM model architecture
class BERTLSTMModel(torch.nn.Module):
    def __init__(self, bert_model_name, max_emojis, embedding_size, lstm_hidden_size):
        super(BERTLSTMModel, self).__init__()

        # BERT model for text
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(0.3)

        # Linear layer to project BERT output to match LSTM output
        self.bert_to_lstm = torch.nn.Linear(self.bert.config.hidden_size, lstm_hidden_size)

        # LSTM model for emoji embeddings
        self.emoji_embedding = torch.nn.Embedding(num_embeddings=max_emojis, embedding_dim=embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)

        # Learnable weights for text and emoji contributions
        self.text_weight = torch.nn.Parameter(torch.tensor(0.85))  # Emphasize text more
        self.emoji_weight = torch.nn.Parameter(torch.tensor(0.15))  # De-emphasize emoji contribution

        # Regressor for predicting score out of 5 (single output)
        self.regressor = torch.nn.Linear(lstm_hidden_size, 1)

        # Sigmoid activation to scale output between 0 and 5
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, emoji_indices):
        # BERT embeddings for text
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled_output = bert_outputs.pooler_output
        bert_pooled_output = self.dropout(bert_pooled_output)

        # Project BERT output to LSTM size
        bert_pooled_output = self.bert_to_lstm(bert_pooled_output)

        # Emoji LSTM embeddings
        emoji_embeds = self.emoji_embedding(emoji_indices)
        lstm_out, _ = self.lstm(emoji_embeds)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM

        # Weighted combination of text and emoji embeddings
        combined_output = self.text_weight * bert_pooled_output + self.emoji_weight * lstm_out

        # Predict review score
        score = self.regressor(combined_output)

        # Scale the output between 0 and 5 using sigmoid
        score = self.sigmoid(score) * 5  # Scale output between 0 and 5

        return score

# Update max_emojis to match the checkpoint's size (223)
embedding_size = 50
lstm_hidden_size = 50
max_emojis = 223  # Update this to match the size of the emoji embedding layer in your checkpoint

# Initialize the model
model = BERTLSTMModel('bert-base-uncased', max_emojis=max_emojis, embedding_size=embedding_size, lstm_hidden_size=lstm_hidden_size)

# Load the model's state dictionary
model_save_path = "bert_lstm_model2.pth"  # Update with your model path
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))  # Load weights
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_save_path}")

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the device

# Example preprocess function (replace with your actual tokenization process)
def preprocess_single_review(review, emoji, emoji_data):
    # Example token IDs (replace with your actual tokenization process)
    input_ids = torch.tensor([1, 2, 3])  # Example token IDs
    attention_mask = torch.tensor([1, 1, 1])  # Example attention mask

    # Add batch dimension (batch_size=1)
    input_ids = input_ids.unsqueeze(0)  # Shape: (1, sequence_length)
    attention_mask = attention_mask.unsqueeze(0)  # Shape: (1, sequence_length)

    # Example emoji index (replace with actual emoji encoding)
    emoji_index = torch.tensor([emoji_data.get(emoji, 0)]).unsqueeze(0)  # Shape: (1, 1)

    return input_ids, attention_mask, emoji_index

# Define the prediction endpoint
@app.post("/predict")
async def predict_sentiment(data: ReviewInput):
    review = data.review
    emoji = data.emoji

    try:
        # Preprocess the review and emoji
        input_ids, attention_mask, emoji_index = preprocess_single_review(review, emoji, emoji_data)

        # Move inputs to the correct device (CPU or GPU)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        emoji_index = emoji_index.to(device)

        # Forward pass through the model
        with torch.no_grad():
            predicted_score = model(input_ids, attention_mask=attention_mask, emoji_indices=emoji_index)
        predicted_score = predicted_score.squeeze().item()

        # Apply penalty for negative words in the review
        for word in negative_words:
            if word in review.lower():
                predicted_score -= 0.5  # Deduct a fixed amount for negative words

        # Apply penalty for negative emojis
        if emoji in negative_emojis:
            predicted_score -= 0.5  # Deduct for negative emoji

        # Ensure the score stays between 0 and 5
        predicted_score = max(0, min(predicted_score, 5))

        return {"review": review, "emoji": emoji, "predicted_score": round(predicted_score, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
