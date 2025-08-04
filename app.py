import streamlit as st
from src.sentiment_logic import predict_sentiment

# --- App Config ---
st.set_page_config(page_title="Sentiment-Aware Chatbot", page_icon="🤖")
st.title("🤖 Sentiment-Aware Chatbot")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Response Generator ---
def generate_response(sentiment, confidence=None):  # confidence kept but unused
    if sentiment == 'positive':
        return "😊 I'm glad to hear that!", "green"
    elif sentiment == 'neutral':
        return "😐 Thanks for sharing your thoughts.", "gray"
    elif sentiment == 'negative':
        return "😟 I'm sorry to hear that. Let me know if I can help!", "red"
    else:
        return "🤔 I'm not sure how to respond.", "blue"

# --- Chat Input UI ---
user_input = st.text_input("Type your message here...")

if user_input:
    sentiment, confidence = predict_sentiment(user_input)
    reply, color = generate_response(sentiment)

    # Save chat
    st.session_state.messages.append(("You", user_input, "#0d6efd"))
    st.session_state.messages.append((f"Bot ({sentiment})", reply, color))

# --- Display Chat History ---
st.markdown("### 💬 Chat History")
for sender, message, color in st.session_state.messages:
    st.markdown(
        f"<p style='color:{color}'><b>{sender}:</b> {message}</p>",
        unsafe_allow_html=True
    )