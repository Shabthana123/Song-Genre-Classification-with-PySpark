import streamlit as st
from classify_logistic import predict_and_plot

st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎶 Music Genre Classifier")

st.markdown("Enter some lyrics and predict the most likely music genre:")

# User input
input_lyrics = st.text_area("🎤 Enter Lyrics", height=250, placeholder="Type or paste lyrics here...")

if st.button("Predict Genre"):
    if input_lyrics.strip():
        genre, fig = predict_and_plot(input_lyrics, show_plot=True)
        st.success(f"🎧 **Predicted Genre:** {genre}")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter some lyrics before predicting.")

# run streamlit app with:
# streamlit run app.py