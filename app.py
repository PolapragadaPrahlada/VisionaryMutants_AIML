import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to filter jobs based on similarity
def filter_jobs(predicted_label, input_file="all_jobs.csv", threshold=0.6):
    df = pd.read_csv(input_file)
    df['combined_text'] = df['Job Title'] + " " + df['Role'] + " " + df['skills']
    label_embedding = model.encode(predicted_label, convert_to_tensor=True)

    similarities = [
        util.cos_sim(label_embedding, model.encode(text, convert_to_tensor=True)).item()
        for text in df['combined_text']
    ]
    df['similarity'] = similarities
    filtered_jobs = df[df['similarity'] >= threshold]
    return filtered_jobs.sort_values(by='similarity', ascending=False)

# Streamlit app
def main():
    st.title("AI-Driven Resume Matcher")

    # Step 1: Upload Resume
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a text file with your resume", type="txt")

    if uploaded_file is not None:
        # Read the uploaded file
        resume_text = uploaded_file.read().decode("utf-8")
        st.write("Resume Preview:")
        st.text(resume_text)

        # Step 2: Predict Job Role (Replace with your model logic)
        predicted_label = "Java Developer"  # Example static prediction
        st.write(f"Predicted Job Role: **{predicted_label}**")

        # Step 3: Filter Jobs
        if st.button("Find Relevant Jobs"):
            with st.spinner("Finding relevant jobs..."):
                filtered_jobs = filter_jobs(predicted_label)

            # Step 4: Display Filtered Jobs
            st.header("Filtered Job Results")
            if not filtered_jobs.empty:
                st.dataframe(filtered_jobs[['Job Title', 'Role', 'skills', 'similarity']])
                # Option to download the results
                csv = filtered_jobs.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="filtered_jobs.csv",
                    mime="text/csv"
                )
            else:
                st.write("No matching jobs found!")

if __name__ == "__main__":
    main()
