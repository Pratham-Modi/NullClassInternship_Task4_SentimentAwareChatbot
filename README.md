# 🤖 Task 4: Sentiment-Aware Chatbot

This project is an intelligent, real-time **Sentiment-Aware Chatbot** that detects emotions from user input and responds with empathetic, context-appropriate replies. Built with **Streamlit**, it leverages a machine learning pipeline trained on **15,000 high-quality samples** and supports **positive**, **neutral**, and **negative** sentiment detection.

---

## 🚀 Tech Stack

- 🎨 **Streamlit** – Interactive chatbot interface  
- 🧠 **Scikit-learn** – TF-IDF + Logistic Regression for sentiment classification  
- 🧹 **NLTK** – Lemmatization and preprocessing  
- 📊 **Seaborn/Matplotlib** – Confusion matrix for evaluation  
- 🧠 **Custom dataset** – 15k samples with rich tone diversity  
- 🧾 **Pickle** – Model + Vectorizer persistence  

---

## ✨ Features

✅ Real-time chatbot with sentiment-based dynamic responses  
✅ Handles subtle expressions, sarcasm, and nuanced tones  
✅ Clean UI with color-coded replies, emoji support, and chat history  
✅ 15K-line dataset 
✅ Robust text preprocessing (stopwords, lemmatization, n-grams)  
✅ Confidence-aware response logic (hidden from UI)  
✅ Easy to retrain with new data  

---

## 🧱 Project Structure

```
Task4_SentimentChatbot/
│
├── data/
│   └── sentiment_dataset.csv        # Final 15,000-line dataset for training
│
├── src/
│   └── sentiment_logic.py           # Preprocessing and prediction logic used in app
│
├── sentiment_model.ipynb            # Jupyter notebook for model training, evaluation, and prediction
├── sentiment_model.pkl              # Saved trained model, vectorizer, label encoder
├── app.py                          # Streamlit UI and chatbot logic
├── requirements.txt                # Dependencies
├── .gitignore                     # Ignored files
└── README.md                      # Project overview and instructions
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Pratham-Modi/NullClassInternship_Task4_SentimentAwareChatbot
cd NullClassInternship_Task4_SentimentChatbot
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 💬 Sample Interactions

**You:** I didn't expect this to be so good!  
**Bot (positive):** 😊 I'm glad to hear that!

**You:** It’s alright, could be better.  
**Bot (neutral):** 😐 Thanks for sharing your thoughts.

**You:** Worst experience ever. Not happy at all.  
**Bot (negative):** 😟 I'm sorry to hear that. Let me know if I can help!

---

## 📊 Model Details

- **Algorithm:** Logistic Regression  
- **Vectorizer:** TF-IDF with n-grams (1,2), stopword removal, sublinear TF  
- **Accuracy:** ~85% on test set (clean, balanced data)  
- **Dataset:** 15,000 entries with realistic tones (neutral/contrastive/sarcastic)  
- **Preprocessing:**
  - Stopword removal (via `sklearn`)
  - Lemmatization (via `nltk`)
  - Cleaned punctuation, links, and case

---

## 📦 requirements.txt

```
streamlit
scikit-learn
nltk
seaborn
matplotlib
```

---

## 📌 Notes

- The model uses a confidence score to guide internal logic (not shown in UI)
- All data resides locally — no API calls required
- Easily extendable: just add more rows to `data/sentiment_dataset.csv` and retrain

---

## 👨‍💻 Developed By

**Pratham Modi**  
📅 July 2025 
