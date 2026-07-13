
import streamlit as st
from transformers import pipeline 
import requests
from bs4 import BeautifulSoup



#function to fetch text from url

def fetch_text_from_url(url:str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    soup =BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    if not paragraphs:
        raise ValueError("No readable text found on this webpage.")
    
    return"\n\n".join(paragraphs)


#function to clean +truncate text 

def clean_and_truncate(text:str, max_chars :int =3500)->str:

    cleaned = " ".join(text.strip().split())
    if len(cleaned)>max_chars:
        cleaned= cleaned [:max_chars]+"..."
    return cleaned 

class Summarizer:
    
    def __init__(self, model_name: str="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.pipeline= pipeline("summarization",model=model_name)
        
    def summarize(self, text:str, max_length:int= 150, min_length:int=40,do_sample:bool=False)->str:
        result = self.pipeline(text, max_length= max_length ,min_length=min_length, do_sample=do_sample)
        return result[0].get("summary_text","")
    
    
# streamlit UI
@st.cache_resource
def load_summarizer(model_name):
    return Summarizer(model_name)

def main():
    st.set_page_config(page_title="Text Summarizer App",layout='wide')
    st.title("Text Summarizer App - Python + HuggingFace")
    st.markdown("summarize text,files and webpages")
    
    left,right =st.columns([1,2])
    
    with left:
        input_mode =st.radio("Input Type: ",["Text","File","URL"])
        model_choice =st.selectbox("Model:",
            ["sshleifer/distilbart-cnn-12-6","facebook/bart-large-cnn","t5-small"])
        
        min_length =st.number_input("Minimum Summary Tokens:", min_value=5 , max_value=100, value =40)
        max_length =st.number_input("Maximum Summary Tokens:", min_value=10 , max_value=2000, value=150)
        
        
        do_sample = st.checkbox("Use Sampling (creative summaries)", value=False)
        
        raw_text = None
        uploaded_file = None
        url_input = None 
        
        
        if input_mode =="Text":
            raw_text = st.text_area("Enter text here:",height=250)
        elif input_mode =="File":
            uploaded_file = st.file_uploader("upload.txt or.md file",type=["txt", "md"])
            
        elif input_mode=="URL":
            url_input =st.text_input("Enter webpage URL");
            
        Summarize_btn = st.button("summarize")
        
        
        
    with right :
        st.subheader("Original text")
        source_container = st.empty()
        
        st.subheader("summary output")
        summary_container = st.empty()
        
        
        if Summarize_btn:
            
            user_input_text = ""
            
            if input_mode == "Text":
                if raw_text:
                    user_input_text = raw_text
                    
            elif input_mode == "File":
                if uploaded_file is not None:
                    uploaded_bytes = uploaded_file.read()
                    try:
                        user_input_text = uploaded_bytes.decode("utf-8")
                    except:
                        user_input_text = uploaded_bytes.decode("latin-1")
            
            elif input_mode == "URL":     
                if url_input:
                    try:
                        user_input_text = fetch_text_from_url(url_input)
                    except Exception as e:
                        st.error(f"Failed to fetch URL: {e}")
                        return
    
            if not user_input_text.strip():
                st.warning("Please provide valid input.")
                return
            user_input_text = clean_and_truncate(user_input_text)
        
            source_container.code(user_input_text[:8000])
        
            if min_length>= max_length:
                st.error("Minimum tokens must be smaller than Maximum tokens:")
                return
        
            summarizer = load_summarizer(model_choice)
    
            with st.spinner("Generating summary..."):
                try:
                    summary_text= summarizer.summarize(
                        user_input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample
                    )
                except Exception as e:
                    st.error(f"summarization failed : {e}")
                    return

            if not summary_text.strip():
                st.warning("The model returned an empty summary ,Try reducing minimum tokens or adding more text.")
                return
            st.success("Summary generated successfully!")
            summary_container.text_area(
                "Summary",
                summary_text,
                height=250
)
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain"
)
    
    
    
if __name__=="__main__":
    main()