import streamlit as st
from PIL import Image
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load models
learn_emotion = load_learner('emotions_vgg19.pkl')
learn_sentiment = load_learner('sentiment_vgg19.pkl')

# Define function for prediction
def predict(image):
    # Open image
    img = Image.open(image)
    
    # Resize image to 48x48
    img = img.resize((48, 48))
    
    # Convert image to grayscale
    img = img.convert('L')
    
    # Convert image to tensor
    img = PILImage.create(img)
    
    # Perform predictions
    pred_emotion, pred_emotion_idx, probs_emotion = learn_emotion.predict(img)
    pred_sentiment, pred_sentiment_idx, probs_sentiment = learn_sentiment.predict(img)
    
    # Convert probabilities to percentages
    probs_emotion = [float(prob) * 100 for prob in probs_emotion]
    probs_sentiment = [float(prob) * 100 for prob in probs_sentiment]
    
    return pred_emotion, pred_sentiment, probs_sentiment, probs_emotion

# Streamlit app
st.title("Facial Expression Classification")
st.markdown(
    """Ever wondered what a person might be feeling looking at their picture? 
    Well, now you can! Try our fun project."""
)

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Perform prediction
    pred_emotion, pred_sentiment, probs_sentiment, probs_emotion = predict(uploaded_image)
    
    # Display results
    st.subheader("Emotion")
    st.write(f"Predicted Emotion: {pred_emotion}")
    st.write("Emotion Probabilities:")
    st.write({
        'Angry': probs_emotion[0],
        'Disgust': probs_emotion[1],
        'Fear': probs_emotion[2],
        'Happy': probs_emotion[3],
        'Sad': probs_emotion[4],
        'Surprise': probs_emotion[5],
        'Neutral': probs_emotion[6]
    })
    
    st.subheader("Sentiment")
    st.write(f"Predicted Sentiment: {pred_sentiment}")
    st.write("Sentiment Probabilities:")
    st.write({
        'Positive': probs_sentiment[0],
        'Negative': probs_sentiment[1],
        'Neutral': probs_sentiment[2]
    })