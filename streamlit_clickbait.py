import torch
import torch.nn.functional as F
import streamlit as st
from transformers import BertTokenizer, BertModel
from bert_sentiment_model import BertClassifier  # Pastikan import model yang sudah dibuat sebelumnya

# Load tokenizer dan model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertClassifier(num_classes=2)
model.load_state_dict(torch.load('bert_sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Sentiment Analysis with BERT")
text = st.text_area("Enter text:", "")

def predict(text):
    label_text = ['Negative', 'Positive']
    encoded = tokenizer.encode_plus(
        text,
        max_length=256,
        padding='max_length',
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    with torch.no_grad():
        out = model(input_ids, attention_mask)
        prob = F.softmax(out, dim=1)
        confidence, predicted_class = torch.max(prob, dim=1)
        predicted_class = predicted_class.item()
        prob = prob.flatten().tolist()
        return label_text[predicted_class], confidence.item()

if st.button("Analyze"):
    sentiment, confidence = predict(text)
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Confidence: **{confidence:.4f}**")
    st.write(f"Confidence: **{confidence:.4f}**")
