from flask import Flask, request, jsonify, render_template
import torch
from transformers import RobertaTokenizer

print("Starting the Flask app...")

app = Flask(__name__)

# ✅ Load model
try:
    model = torch.load('model_try_1_cpu.pkl', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    print("Model loaded successfully ✅")
except Exception as e:
    print("Error loading model:", e)

# ✅ Load tokenizer (same as in your training)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# ✅ Define class mapping (index → name)
class_names = {
    0: "Depression",
    1: "Sadness",
    2: "Normal",
    3: "Suicidal",
    4: "Anxiety",
    5: "Bipolar",
    6: "Stress",
    7: "Personality disorder"
}

# ✅ Preprocessing function (like in training)
def preprocess(text_list):
    tokenized = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return tokenized

@app.route('/')
def home():
    return render_template('home.html')  # Optional HTML UI

# ✅ Batch prediction endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']  # Expecting a list of texts
        if isinstance(data, str):
            data = [data]
        # Tokenize
        tokens = preprocess(data)
        # Predict
        with torch.no_grad():
            outputs = model(**tokens)
            predictions = torch.argmax(outputs.logits, dim=1)
        # Map to class names
        predicted_classes = [class_names[p.item()] for p in predictions]
        return jsonify(predicted_classes)
    except Exception as e:
        return jsonify({"error": str(e)})
    
# ✅ Single text prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data["text"]
        # Tokenize and predict using your model
        tokens = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        prediction_name = class_names[prediction]
        return jsonify({"prediction": prediction_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5050)
