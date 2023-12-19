import numpy as np
import streamlit as st
import pdfplumber
import re
import base64

import os
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
model = Doc2Vec.load('model/doc2vec_v4en.model')

# Fixed dataset of open positions
jobs = pd.read_csv('data/hhparser_vacancy_short.csv')

# Embeddings
index = read_index("model/hh_v4en.index")


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

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2)
            if text:
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                resume.append(text)

    resume = ' '.join([e.replace('\n', ' ').lower() for e in resume])
    # print(resume)
    return resume


def weighted_read_resume(path) -> str:
    """
    Reading file and multiplying words based on relative font size
    May work better since Doc2Vec uses DBOW which does a bag-of-words style architecture

    :param str path: path to file
    :return: processed str
    """

    font_sizes = []
    weighted_text = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, keep_blank_chars=True, use_text_flow=True,
                                       extra_attrs=["size"])
            font_sizes.extend([float(word['size']) for word in words])

        # Calculate median font size
        median_size = np.median(font_sizes)

        # a bit of processing - some non-letter chars can be considered as spaces

        for word in words:
            word_text = word['text'].replace('/', ' ').replace('-', ' ').strip('.,').lower()
            word_size = float(word['size'])

            # Assign weight based on font size (example: 2x for each size unit above median)
            if word_size > median_size:
                weight = (word_size * 1.3 / median_size)
            else:
                weight = 1

            # Replicate word based on weight
            weighted_text.extend([word_text] * int(round(weight)))

    weighted_text = " ".join([e for e in weighted_text if any(c.isalpha() for c in e)])

    return weighted_text


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
        st.header("Results", anchor=False)

        if st.session_state['file_loaded']:
            # read and process file
            resume_vocab = read_resume(f"cache/{uploaded_file.name}")

            # get doc vector
            v1 = np.array([model.infer_vector(resume_vocab.split())])

            # find the closest embedding in index
            distances, indices = index.search(v1, 5)

            # print(indices)

            matches = []
            for i in indices[0]:
                title = f"[{jobs.loc[i, 'name']} - {jobs.loc[i, 'employer_name']}]({jobs.loc[i, 'alternate_url']})"

                if jobs.loc[i, ['salary_from', 'salary_to']].notna().all():
                    salary = f"{int(jobs.loc[i, 'salary_from'])} - {int(jobs.loc[i, 'salary_to'])}"
                elif jobs.loc[i, 'salary_from'] > 0:
                    salary = f"З/П от {int(jobs.loc[i, 'salary_from'])}"
                elif jobs.loc[i, 'salary_to'] > 0:
                    salary = f"З/П до {int(jobs.loc[i, 'salary_to'])}"
                else:
                    salary = f"З/П не указана"

                descriprion = jobs.loc[i, 'description'] if jobs.loc[i, 'description'] \
                    else "Описание доступно только по ссылке :("

                matches.append(f"## {title} \n"
                               f"### {salary} \n"
                               f"{descriprion}")

            if matches:
                st.markdown(f"## It's a match! \n", unsafe_allow_html=True)

                # Pagination UI
                pagination_container = st.container()
                prev, next, counter = pagination_container.columns([1, 1, 8])

                if prev.button("Previous"):
                    st.session_state['current_page'] = max(0, st.session_state['current_page']-1)
                if next.button("Next"):
                    st.session_state['current_page'] = min(len(matches)-1, st.session_state['current_page']+1)
                counter.markdown(f"#### Page {st.session_state['current_page'] + 1} of {len(matches)}")

                # Determine the offers to display on the current page
                match = matches[st.session_state['current_page']]
                st.markdown(match, unsafe_allow_html=True)


if __name__ == "__main__":

    # Create / clear cache
    if "cache" not in os.listdir(os.curdir):
        os.mkdir("cache")
    else:
        for file in os.listdir("cache/"):
            os.unlink(f"cache/{file}")

    main()
