# **************************************************************************
# *
# * Authors: Alberto Garcia Mena   (alberto.garcia@cnb.csic.es)
# *
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import argparse
from langchain_huggingface import HuggingFaceEmbeddings
import faiss #faiss-cpu
import numpy as np
import json

######CONSTANTS
SIZE_USER_QUESTION = 100
INDEX_VECTOR_DIMENSION = 768
NPY_FILE = 'indexMap.npy'
FAISS_FILE= 'indexMap.faiss'
JSON_MAP = 'indexMap.json'
VECTORS_SEARCHED = 15
MINIMUM_CORRELATION_REQUIRED = 0.3
#####CONFIGURATIONS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def parseUserQuestion():
    parser = argparse.ArgumentParser(description='Ejemplo arg')
    parser.add_argument('userQuestion', type=str, help='question to recomend a protocol')

    args = parser.parse_args()
    userQuestion = args.userQuestion
    if userQuestion is None:
        userQuestion = 'a protocol to align movies'

    if len(userQuestion) > SIZE_USER_QUESTION:
        print(f'the size of the question is larger than {SIZE_USER_QUESTION}')

    return userQuestion



def embedUserQuestion(embedUserQuestion):
    userQuestionVector = embeddings.embed_query(embedUserQuestion)
    return np.array(userQuestionVector).reshape(1, -1)


def searchOnIndexFaiss(userQuestionVector):
	indexFaiss = faiss.read_index(FAISS_FILE)
	return indexFaiss.search(userQuestionVector, k=VECTORS_SEARCHED)


def evaluateCorrelations(correlation, index):
	#Save and sort by correlation a dictCorrIndex
	dictCorrIndex = {}
	for i in range(VECTORS_SEARCHED):
		if correlation[0, i] == -1 or correlation[0, i] < MINIMUM_CORRELATION_REQUIRED:
			continue
		dictCorrIndex[index[0, i]] = correlation[0, i]

	dictCorrIndex = dict(sorted(dictCorrIndex.items(), key=lambda item: item[1], reverse=True))
	return dictCorrIndex


def findProtocolsRecomended(dictCorrIndex):
	with open(JSON_MAP, "r", encoding="utf-8") as file:
		dictMap = json.load(file)
	dictProtocolcorr = {}
	for i, (key, value) in enumerate(dictCorrIndex.items()):
		dictProtocolcorr[i] = {'PLUGIN': dictMap['VECTORS'][str(key)]['PLUGIN'],
							   'PROTOCOL': dictMap['VECTORS'][str(key)]['PROTOCOL'],
							   'CORRELATION': value}
	return  dictProtocolcorr


def printRecomendations(dictProtocolcorr):
	print(f'Protocol recomended: {next(iter(dictProtocolcorr.items()))}')




if __name__ == "__main__":
	userQuestion = parseUserQuestion()
	userQuestionVector = embedUserQuestion(userQuestion)
	correlation, index = searchOnIndexFaiss(userQuestionVector=userQuestionVector)
	dictCorrIndex = evaluateCorrelations(correlation, index)
	if dictCorrIndex:
		dictProtocolcorr = findProtocolsRecomended(dictCorrIndex)
		#collectReportAboutProtocol(dictProtocolcorrSorted)
		printRecomendations(dictProtocolcorr)
	else:
		print(f'None protocol recomended based on the user question:\n {userQuestion}')