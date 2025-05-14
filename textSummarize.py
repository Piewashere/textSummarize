from gpt4all import GPT4All
import PyPDF2
import streamlit as st

# Center the title using HTML
st.markdown("<h1 style='text-align: center;'>PDF Summarizer</h1>", unsafe_allow_html=True)

model = GPT4All("Phi-3-mini-4k-instoruct.Q4_0.gguf", model_type="llama")

st.write("Upload a PDF file to summarize its content.")
uploaded_file = st.file_uploader("")

if uploaded_file:
    if uploaded_file.type != "application/pdf":
        st.error("Please upload a valid PDF file.")
    else:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        prompt = f"Summarize this into three bullet points and a conclusion: \n\n{text}\n\n"

        with st.spinner("Generating summary..."):
            with model.chat_session():
                response = model.generate(prompt, max_tokens=1024)
        
        st.success("Summary generated successfully!")
        st.write("Summary:\n")
        st.write(response)
        st.write("________________________________")