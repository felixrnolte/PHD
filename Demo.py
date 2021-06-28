# This is a demo file that can be executed with a set of example inputs to show the transformation between text and model step by step. The example inputs are taken from the set of models and descriptions used for the evaluation. The inputs corresponding to the entries 1,4,5,9 and 10 of that dataset.

# 0) Preparations

# You have to download the latest StanfordCoreNLP model from https://stanfordnlp.github.io/CoreNLP/index.html#download and call it with the following command, adjusting the path accordingly
#java -mx4g -cp "C:\Users\felix\Downloads\stanford-corenlp-full-2018-10-05\\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

# a) Test Inputs

import pandas

test_demo_1_adv_cond = "Laura reviews the lead activity from a received e-mail. If a recent activity can be identified, she qualifies the lead. If no recent activity is identified, the lead is disqualified. After Laura qualifies the lead, she informs the other departments."

test_demo_2_rel_generation = "Laura creates the list of products that she received. From the list of products, Michael prepares the invoice. He then sends the package."

test_demo_3 = "From an accepted offer, an order is entered. Based on the new order, the items are sent to complete the delivery."

# Set on test case as input for the demo
input_text = test_demo_2_rel_generation

print(input_text)
print("===================================================================================================")
print("======================================= Starting Generation =======================================")
print("===================================================================================================")
# 1) Text Pre-Processing
# a Anaphora Resolution
from Text_Preprocessing.Anaphora_Resolution import get_resolved

proc_text = get_resolved(input_text)
print(proc_text)
print("==========================================================================================================")
print("======================================= Anaphora Resolution - DONE =======================================")
print("==========================================================================================================")
# c) Sentence Splitter
from Text_Preprocessing.Sentence_Splitting import _get_sentences_spacy

sentences = _get_sentences_spacy(proc_text)

print(sentences)
sent_df = pandas.DataFrame(sentences)
sent_df.to_csv(r'Data\Demo\Sentences_' + 'demo.csv', index = False)
print("========================================================================================================")
print("======================================= Sentence Splitter - DONE =======================================")
print("========================================================================================================")

# 2) Linguistic Feature Extraction
# For each sentence now phrases are extracted, keywords of these identified and process relevant information extracted

# a) Phrase Extraction
from Linguistic_Feature_Extraction.Phrase_Extraction import _get_phrases_of_sentences

phrases = _get_phrases_of_sentences(sentences)

print(phrases)
phrases.to_csv(r'Data\Demo\Phrases_' + 'demo.csv', index = False)
print("========================================================================================================")
print("======================================= Phrase Extraction - DONE =======================================")
print("========================================================================================================")

# b) Information Extraction (using also keywords)
from Linguistic_Feature_Extraction.Information_Extraction import _get_ling_information

features = _get_ling_information(phrases)

#features.to_csv(r'Data\features.csv', index=False)

print(features)
features.to_csv(r'Data\Demo\Features_' + 'demo.csv', index = False)
print("============================================================================================================")
print("======================================= Information Extraction- DONE =======================================")
print("============================================================================================================")

# 3) Model Element Mapping
from Model_Element_Mapping.Mapping import model_element_mapping

mapping_df = model_element_mapping(features)

print("==============================================================================================")
print("==============================================================================================")

print(mapping_df)

mapping_df.to_csv(r'Data\Demo\Mappign_' + 'demo.csv', index = False)

print("==============================================================================================")
print("======================================= Mapping - DONE =======================================")
print("==============================================================================================")
# 4) Model Generation

from Model_Generation.Generation import _connect_parts

model = _connect_parts(mapping_df)

print(model)
model.to_csv(r'Data\Demo\Model_' + 'demo.csv', index = False)
print("=================================================================================================")
print("======================================= Generation - DONE =======================================")
print("=================================================================================================")