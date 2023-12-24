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

import fasttext
import fasttext.util
from sklearn.neighbors import NearestNeighbors
from joblib import load

# Initialize session state variables if they don't exist
if 'file_loaded' not in st.session_state:
    st.session_state['file_loaded'] = False

# Initialize or update the current page in session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0

# Model weights
model = Doc2Vec.load('model/doc2vec_v4en.model')

# Fixed dataset of open positions
jobs = pd.read_csv('data/edited.csv')

# Embeddings
index = read_index("model/hh_v4en.index")

#knn Model
knn_model = load('model/knn.joblib')

#fasttext model
os.chdir('model')
fasttext.util.download_model('ru', if_exists='ignore')
os.chdir('..')
fasttext_model = fasttext.load_model('model/cc.ru.300.bin')
fasttext.util.reduce_model(fasttext_model, 100)


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

def get_short_jobs(data, specialty, region, expected_salary):
    vacancy_emb = [fasttext_model.get_word_vector(word) for word in specialty.split(' ')][0].tolist()
    my_city_emb = [fasttext_model.get_word_vector(word) for word in region.split(' ')][0].tolist()  
    res = knn_model.kneighbors([vacancy_emb+my_city_emb+[expected_salary]], return_distance=False)
    data = data.iloc[res[0]].copy()
    return data

def main():
    # Page config
    st.set_page_config(page_title="CV Matching App",
                       page_icon="🧊",
                       layout="wide")
    st.title("🔮 Resume matching app 🔮", anchor=False)

    # Splitting the page into two columns
    col1, col_mid, col2 = st.columns([6, 0.4, 6])

    with col1:
        st.header("Upload Resume", anchor=False)

        with st.form("CV_upload_form", clear_on_submit=False, border=False):
            # Список введеных пользователем полей: специальность, регион, образование
            # и чуть ниже мин. и макс. ожидаемая з/п

            specialty = st.text_input("Введите вашу специальность", value="")
            region = st.text_input("Введите регион", value="")
            #А нужен ли нам этот эдукейшон ваш? 
            education = st.text_input("Введите свой текущий уровень образования", value="")
            expected_salary = st.number_input("Введите свою ожидаемую з/п",
                                              min_value=0,
                                              step=5000)

            # Пусть будет допустим диапазон ±20% от введенной з/п
            min_expected_salary, max_expected_salary = 0.8 * expected_salary, 1.2 * expected_salary

            uploaded_file = st.file_uploader("Upload your CV in PDF format*", type="pdf", help='Обязательное поле')
            submitted = st.form_submit_button("Отправить CV")
        if submitted is not None:
            if uploaded_file:
                # Display the PDF
                with open(f"cache/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # st.success("File Uploaded Successfully")
                st.session_state['file_loaded'] = True

                displayPDF(f"cache/{uploaded_file.name}")
            else:
                st.info("Для того, чтобы получить результат, необходимо загрузить PDF-файл резюме и нажать на кнопку")
                st.session_state['file_loaded'] = False

    with col2:
        if st.session_state['file_loaded']:
            st.header("Results", anchor=False)
            # read and process file
            resume_vocab = read_resume(f"cache/{uploaded_file.name}")

            # get doc vector
            v1 = np.array([model.infer_vector(resume_vocab.split())])

            # find the closest embedding in index
            distances, indices = index.search(v1, 5)

            # print(indices)

            matches = []
            jobs = get_short_jobs(jobs, specialty, region, expected_salary)
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
                prev, next, counter = pagination_container.columns([2, 2, 6])

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
