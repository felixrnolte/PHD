import streamlit as st
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")

# --------------------------------
# Sidebar
#---------------------------------
st.sidebar.header('Set Process Description')

test_demo_1_adv_cond = "Laura reviews the lead activity from a received e-mail. If a recent activity can be identified, she qualifies the lead. If no recent activity is identified, the lead is disqualified. After Laura qualifies the lead, she informs the other departments."

test_demo_2_rel_generation = "Laura creates the list of products that she received. From the list of products, Michael prepares the invoice. He then sends the package."

def user_input_features():
    Input_Text = st.sidebar.text_input("Process Description", test_demo_1_adv_cond)

    data = {
        "Process_Description": Input_Text
    }

   
    return Input_Text

df = user_input_features() # Data from the sidebar controls 

# --------------------------------

st.write(
    '''
    # Demo Case
    '''
)

# 1) Text Pre-Processing
# a Anaphora Resolution

st.subheader('Process Description')

st.write(df)
print(df)

st.subheader('Coreference/Anaphora Resolution')

from Text_Preprocessing.Anaphora_Resolution import get_resolved


proc_text = get_resolved(df)
st.write(proc_text)
# c) Sentence Splitter
st.subheader('Sentence Splitting')

from Text_Preprocessing.Sentence_Splitting import _get_sentences_spacy

sentences = _get_sentences_spacy(proc_text)

sent_df = pandas.DataFrame(sentences)
#sent_df.to_csv(r'Data\Demo\Sentences_' + 'demo.csv', index = False)
st.write(sent_df)

# 2) Linguistic Feature Extraction
# For each sentence now phrases are extracted, keywords of these identified and process relevant information extracted

# a) Phrase Extraction

st.subheader('Phrase Extraction')

from Linguistic_Feature_Extraction.Phrase_Extraction import _get_phrases_of_sentences

phrases = _get_phrases_of_sentences(sentences)

#phrases.to_csv(r'Data\Demo\Phrases_' + 'demo.csv', index = False)
st.dataframe(phrases, width=1500, height=300)
#st.write(phrases)

# b) Information Extraction (using also keywords)
from Linguistic_Feature_Extraction.Information_Extraction import _get_ling_information

st.subheader('Linguistic Information Extraction')

features = _get_ling_information(phrases)
st.dataframe(features, width=1500, height=300)
#st.write(features)
#features.to_csv(r'Data\features.csv', index=False)

#features.to_csv(r'Data\Demo\Features_' + 'demo.csv', index = False)


# 3) Model Element Mapping
st.subheader('Model Element Mapping')

from Model_Element_Mapping.Mapping import model_element_mapping

mapping_df = model_element_mapping(features)
st.dataframe(mapping_df, width=1500, height=300)
#st.write(mapping_df)

#mapping_df.to_csv(r'Data\Demo\Mappign_' + 'demo.csv', index = False)

# 4) Model Generation

from Model_Generation.Generation import _connect_parts

st.subheader('Model Generation')

model = _connect_parts(mapping_df)
st.dataframe(model, width=1500, height=300)
#st.write(model)

#model.to_csv(r'Data\Demo\Model_' + 'demo.csv', index = False)







