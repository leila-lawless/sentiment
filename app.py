import os  # <-- ADD THIS LINE
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load model directly from Hugging Face Hub
model_name = "leila-may/sentiment-bot-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process output
        prediction = torch.argmax(outputs.logits).item()
        return jsonify({"sentiment": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Properly get port first
    app.run(host='0.0.0.0', port=port)
