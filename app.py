import streamlit as st
import joblib

logistic_regression_model = joblib.load("/Users/stefaniehegele/Desktop/decision-dill-student-code/Final Project/Hate-Speech-Detector/logistic_regression_model.pkl")
vectorizer = joblib.load("/Users/stefaniehegele/Desktop/decision-dill-student-code/Final Project/Hate-Speech-Detector/vectorizer.pkl")

labels = ["This is a neutral sentence", "This sentence likely contains hate speech"]

def main():
    st.markdown("<h1 style='text-align: center;'>Hate Speech Detection App</h1>", unsafe_allow_html=True)
    
    # Create a column layout with two columns
    col1, col2 = st.columns([2, 1])  # Adjust the column widths as needed

    # Add the text input field in the first column
    with col1:
        user_input = st.text_input("Enter a sentence:")

        if st.button("Predict"):
            processed_test_sentence = user_input.strip().lower()

            if processed_test_sentence != "":
                test_vec = vectorizer.transform([processed_test_sentence])

                prediction = logistic_regression_model.predict(test_vec)

                label = labels[prediction[0]]  # Convert prediction to integer index

                st.write("Prediction:", label)

    # Add the photo in the second column
    with col2:
        st.image("hate love.jpg", caption="Photo by Dan Edge on Unsplash", width=200)


if __name__ == "__main__":
    main()
