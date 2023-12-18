import numpy as np
import streamlit as st
from pathlib import Path
import PyPDF2
import base64

from io import StringIO
from html.parser import HTMLParser

from gensim.models.doc2vec import Doc2Vec
from faiss import read_index
import pandas as pd


# Initialize session state variables if they don't exist
if 'file_loaded' not in st.session_state:
    st.session_state['file_loaded'] = False

# Initialize or update the current page in session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0

# Model weights
model = Doc2Vec.load('model/doc2vec_v2.model')

# Fixed dataset of open positions
jobs = pd.read_csv('data/hhparser_vacancy.csv')

# Embeddings
index = read_index("model/hh.index")


# HTML stripping (https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python)
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_resume(path):
    resume = []

    with open(path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            text = page.extract_text()
            resume.append(text)

    resume = ' '.join([e.replace(' ', '').replace('\n', ' ').lower() for e in resume])
    return resume


def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = (F'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                   F'width="100%" height="1000" type="application/pdf"></iframe>')

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    # Page config
    st.set_page_config(layout="wide")
    st.title("🔮 Resume matching app 🔮", anchor=False)

    # Splitting the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Resume", anchor=False)
        uploaded_file = st.file_uploader("Upload your CV in PDF format...", type="pdf")
        if uploaded_file is not None:
            # Display the PDF
            with open(f"cache/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            # st.success("File Uploaded Successfully")
            st.session_state['file_loaded'] = True

            displayPDF(f"cache/{uploaded_file.name}")

        else:
            st.session_state['file_loaded'] = False

    with col2:
        st.header("Matching Results", anchor=False)

        if st.session_state['file_loaded']:
            # read and process file
            resume_vocab = read_resume(f"cache/{uploaded_file.name}")

            # get doc vector
            v1 = np.array([model.infer_vector(resume_vocab.split())])

            # find the closest embedding in index
            distances, indices = index.search(v1, 5)

            indices = indices[0]

            matches = {}
            for i in set(indices):
                title = f"[{jobs.loc[i, 'name']} - {jobs.loc[i, 'employer_name']}]({jobs.loc[i, 'alternate_url']})"

                salary = f"{jobs.loc[i, 'salary_from']} - {jobs.loc[i, 'salary_to']}" \
                    if jobs.loc[i, ['salary_from', 'salary_to']].notna().all() else "з/п не указана"

                descriprion = jobs.loc[i, 'description'] if jobs.loc[i, 'description'] \
                    else "Описание доступно только по ссылке :("

                matches[i] = (f"## {title} \n"
                              f"### {salary} \n"
                              f"{descriprion}")
                # TODO button gallery with first k (5) matches

            if matches:
                st.markdown(f"## It's a match! \n", unsafe_allow_html=True)

                # Determine the offers to display on the current page
                match = list(matches.values())[st.session_state['current_page']]
                st.markdown(match, unsafe_allow_html=True)

                # Pagination UI
                pagination_container = st.container()
                counter, prev, next = pagination_container.columns([8, 1, 1])

                counter.text(f"Page {st.session_state['current_page'] + 1} of {len(matches)}")

                if prev.button("Previous"):
                    if st.session_state['current_page'] > 0:
                        st.session_state['current_page'] -= 1
                if next.button("Next"):
                    if st.session_state['current_page'] < len(matches) - 1:
                        st.session_state['current_page'] += 1


if __name__ == "__main__":
    main()
