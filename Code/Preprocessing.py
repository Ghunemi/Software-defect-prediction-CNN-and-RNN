import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction import stop_words
import nltk
from nltk.corpus import stopwords
import javalang

"from parser import Parser"
"from tokenizer import tokenize"
stop_words = [';', '@', '(', ')', '{', '}', '*', ',', '/']


class Parser:
    __slots__ = ['name']  # Faster attribute access

    def __init__(self, name):  # Constructor
        self.name = name

    def pre_processing(self):  # pre-processing function

        AST = None
        src = open(self.name, 'r')

        # loop to parse each source code
        for x in range(1):

            src = src.read()

            attributes = []
            variables = []

            # Source parsing
            try:
                AST = javalang.parse.parse(src)  # This will return AST
                for path, node in AST:  # Index, Element
                    if 'ReferenceType' != node:
                        AST.remove(node)
                    print(node, "\n")
                    # print(path,"\n")
            except:
                pass

        vectorizer = TfidfVectorizer(stop_words='english')  # Create the vectorize/transform

        vectorizer.fit([str(AST)])  # Learns vocab " CompilationUnit, Imports, path, static, true, util, io "

        print('---------------------------check 2----------------------------------')
        print(vectorizer.vocabulary_)
        print("STOPPPPING WORDS", vectorizer.get_stop_words())
        vector = vectorizer.transform([str(AST)])  # transform document to matrix
        print(vector)
        print('---------------------check 3-------------------------------------------------------------')
        a = np.array(vector.toarray())
        print(a)
        print('---------------------check 4-------------------------------------------------------------')
        df = DataFrame(a)
        print(df)
       # print("Features")
       # print(vectorizer.get_feature_names())
        df.to_csv('featuresExtraction.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    parser = Parser('')  # Create object of the class parser

    filesNames = pd.read_csv('defectprediction.csv')  # Read all files names
    path = filesNames.iloc[:, 1]  # select column 1, all rows
    filesNames = filesNames.iloc[:, 0]  # select column 0, rows from 0 to length
    filesNames = np.array(filesNames)  # Convert to numpy array
    path = np.array(path)  # Convert to numpy array
    foundsrc = 0
    notdound = 0
    fileNum = 1
    for i in range(path.shape[0]):
        fileNum+=1
        #print("CURRENT\m")
        #print(filesNames[i]," File number: ", fileNum)
        try:
            fh = open(path[i] + ".java", 'r')
            foundsrc += 1  # Increment found counter
            parser = Parser(path[i] + ".java")
            parser.pre_processing()

        except FileNotFoundError:
            notdound += 1  # Increment not found counter
            data = {'Name': [filesNames[i]],
                    'Path': [path[i]],
                    'Status': ['NotFound']}
            df = DataFrame(data)  # add them to data frame
            df.to_csv('NotFoundDefectPrediction.csv', mode='a', index=False, header=False)  # Write missing files in csv

    print("\n\nfound\t\t", foundsrc)
    print("notfound\t", notdound)
