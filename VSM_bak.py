import re
import math
import os
import zipfile
import sys  # get input
import string
from operator import itemgetter
import argparse
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

doc_regex = re.compile("<DOC>.*?</DOC>", re.DOTALL)
docno_regex = re.compile("<DOCNO>.*?</DOCNO>")
text_regex = re.compile("<TEXT>.*?</TEXT>", re.DOTALL)
token_regex = re.compile("\w+([\,\.]\w+)*")
#apply stemming to tokens
porter = PorterStemmer()


#zip extract function, uncomment out later
# with zipfile.ZipFile("ap89_collection_small.zip", 'r') as zip_ref:
#     if "ap89_collection_small" != None:
#         pass
#     else:
#         zip_ref.extractall()
with zipfile.ZipFile("ap89_collection_small.zip", 'r') as zip_ref:
    zip_ref.extractall()

for dir_path, dir_names, file_names in os.walk("ap89_collection_small"):
    allfiles = [os.path.join(dir_path, filename).replace("\\", "/") for filename in file_names if
                (filename != "readme" and filename != ".DS_Store")]


def cosine_similarity(q_id):
        dup_list = [] # stores common token of query and document
        text_sz = 0 # after removing stopping words
        query_sz = 0 # after removing stopping words
        cosine_sim_val = {} # stores all cos_sim values of (q_id , whole ducs)
    #def cosine_similarity(q_id, d_id):

        # q_id = 85
        # d_id = 'AP890101-0001'

        # with open("docids.txt") as did_text:
        #     for line in did_text:
        #         if(d_id == line.split()[1]):
        #             d_id = line.split()[0]
        # print('fixed d_id')

        with open("stopwords.txt", 'r') as sw_text:
            stopwords = sw_text.read().split()
        query_dict = {}  # {"query-number":[], "queries":[]}

        with open('query_list.txt') as q_f:
            query_list = []
            for line in q_f:  # read all lines
                line = line.lower()
                q_num = line.split()[0]
                q_num = q_num[:-1]
                if(q_num == str(q_id)):
                    first = 1
                    query_list = []
                    for token in line.split():
                        if first == 1:
                            first = 0
                        else:
                            stemmed_token = porter.stem(token)
                            if (stemmed_token in stopwords):
                                continue
                            query_sz += 1
                            query_list.append(stemmed_token)
                else:
                    continue


        for file in tqdm(allfiles):
            with open(file, 'r', encoding='ISO-8859-1') as f:
                filedata = f.read()
                result = re.findall(doc_regex, filedata)  # Match the <DOC> tags and fetch documents

                for document in result[0:]:
                    # Retrieve contents of DOCNO tag
                    docno = re.findall(docno_regex, document)[0].replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                    # Retrieve contents of TEXT tag
                    text = "".join(re.findall(text_regex, document)) \
                        .replace("<TEXT>", "").replace("</TEXT>", "") \
                        .replace("\n", " ")

                    text = text.lower()
                    for word in text.split():
                        if word in stopwords:
                            continue
                        text_sz += 1
                        stemmed_word = porter.stem(word)
                        if stemmed_word in query_list:
                            dup_list.append(stemmed_word)
                    cosine_sim_val[docno] = len(set(dup_list)) / (math.sqrt(query_sz) * math.sqrt(text_sz))

        sorted_cosine_sim_val = sorted(cosine_sim_val.items(), key=lambda x: x[1], reverse=True)
        return sorted_cosine_sim_val
#print(len(set(dup_list)), math.sqrt(query_sz), math.sqrt(text_sz))
#print('cosine similarity: ')

#print(cosine_similarity(85)[:10])

def run_query(query_file, output_file):
    with open(query_file, 'r') as q_f, open(output_file, 'w') as output_file:
        for line in q_f:
            rank = 1
            q_id = line.split()[0]
            q_id = q_id[:-1]
            print('q_id : ' + q_id)
            cos_results = cosine_similarity(q_id)[:10]
            print(cos_results)
            for index in cos_results:
                output_file.write(str(q_id) + ' ' + 'Q0' + ' ' + str(index[0]) + ' ' + str(rank) + ' ' + str(index[1]) + ' ' + 'Exp' + '\n')
                rank += 1
            
            # for each docment, run cosine_sim(q_id, doc)
            # show top 10

# <query−number> is the number preceding the query in the query list
# <docno> is the document number, from the< DOCN O >field (which we asked you toindex)
# <rank> is the document rank: an integer from 1-100
# <score> is the retrieval models matching score for the document
# Q0 and Exp are entered literally (because we will use a TREC evaluation code, so the output has to match exactly)
        

#get input ---

#----
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vector Space Model program')
    parser.add_argument('--query', action='store', dest='query_file', type=str, help='Path to query file')
    parser.add_argument('--output', action='store', dest='output_file', type=str, help='Name of output file')
    args = parser.parse_args()

    if args.query_file and args.output_file is not None:
        run_query(args.query_file, args.output_file)
    else:
        # print help if no args
        parser.print_help()

            

            

# open stopwords.txt file and read it
# with open("stopwords.txt", 'r') as sw_text:
#     stopwords = sw_text.read().split()
#
# query_dict = {} #{"query-number":[], "queries":[]}
#
# with open("query_list.txt") as q_f:
#     for line in q_f: #read all lines
#         line = line.lower()
#         q_num = line.split()[0]
#         query_dict[q_num] = []
#         first = 1
#         for token in line.split():
#             if first == 1:
#                 first = 0
#             else:
#                 stemmed_token = porter.stem(token)
#                 if(stemmed_token in stopwords):
#                     continue
#                 query_dict[q_num].append(stemmed_token)
#print(cosine_similarity(85, 1))


