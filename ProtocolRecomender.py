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
import time
from pathlib import Path
import os

start_time = time.time()
######CONSTANTS
PATH_MAP = '/home/agarcia/ProtocolRecomender/'
SIZE_USER_QUESTION = 100
INDEX_VECTOR_DIMENSION = 768
NPY_FILE = 'indexMap.npy'
FAISS_FILE= 'indexMap.faiss'
JSON_MAP = 'indexMap.json'
VECTORS_SEARCHED = 30
MINIMUM_CORRELATION_REQUIRED = 0.3
#####CONFIGURATIONS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

constantEND_time = time.time()
def parseUserQuestion():
    parser = argparse.ArgumentParser(description='Example arg')
    parser.add_argument('userQuestion', type=str, help='question to recomend a protocol')

    args = parser.parse_args()
    userQuestion = args.userQuestion
    if userQuestion == "":
        userQuestion = 'create initial volume'

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
	for i in range(15):
		if i in dictProtocolcorr:
	 		print(f"Protocol: {dictProtocolcorr[i]['PLUGIN']} - {dictProtocolcorr[i]['PROTOCOL']} - {dictProtocolcorr[i]['CORRELATION']}")

def assignScore2Protocols(dictProtocolcorr):
	dictProtocolScore = {}
	for i in range(15):
		if i in dictProtocolcorr:
			if dictProtocolcorr[i]['PROTOCOL'] in dictProtocolScore:
				dictProtocolScore[dictProtocolcorr[i]['PROTOCOL']] +=  dictProtocolcorr[i]['CORRELATION']
			else:
				dictProtocolScore[dictProtocolcorr[i]['PROTOCOL']] =  dictProtocolcorr[i]['CORRELATION']

	#Let's assign one to three starts based on the correlation of the main vectors
	dictProtocolScore = dict(sorted(dictProtocolScore.items(), key=lambda item: item[1]))
	dictProtocolScore = dict(reversed(dictProtocolScore.items()))
	startValue = (VECTORS_SEARCHED - MINIMUM_CORRELATION_REQUIRED) / 10
	starString = '\u2605'
	for p in dictProtocolScore.keys():
		corr = dictProtocolScore[p]
		corrPrint = round(corr, 2)
		if corr > startValue * 9:
			print(f'Protocol: {p} {starString * 10} {corrPrint}')
		elif corr > startValue * 8:
			print(f'Protocol: {p} {starString * 9} {corrPrint}')
		elif corr > startValue * 7:
			print(f'Protocol: {p} {starString * 8} {corrPrint}')
		elif corr > startValue * 6:
			print(f'Protocol: {p} {starString * 7} {corrPrint}')
		elif corr > startValue * 5:
			print(f'Protocol: {p} {starString * 6} {corrPrint}')
		elif corr > startValue * 4:
			print(f'Protocol: {p} {starString * 5} {corrPrint}')
		elif corr > startValue * 3:
			print(f'Protocol: {p} {starString * 4} {corrPrint}')
		elif corr > startValue * 2:
			print(f'Protocol: {p} {starString * 3} {corrPrint}')
		elif corr > startValue:
			print(f'Protocol: {p} {starString * 2} {corrPrint}')
		else:
			print(f'Protocol: {p} {starString } {corrPrint}')



	return dictProtocolScore


if __name__ == "__main__":
	os.chdir(PATH_MAP)
	print(f'Map path: {Path.cwd()}')
	userQuestion = parseUserQuestion()
	parseUserQuestion_time = time.time()
	userQuestionVector = embedUserQuestion(userQuestion)
	embedUserQuestion_time = time.time()
	correlation, index = searchOnIndexFaiss(userQuestionVector=userQuestionVector)
	searchOnIndexFaiss_time = time.time()
	dictCorrIndex = evaluateCorrelations(correlation, index)
	evaluateCorrelations_time = time.time()
	if dictCorrIndex:
		dictProtocolcorr = findProtocolsRecomended(dictCorrIndex)
		findProtocolsRecomended_time = time.time()

		#collectReportAboutProtocol(dictProtocolcorrSorted)
		assignScore2Protocols(dictProtocolcorr)
		#printRecomendations(dictProtocolcorr)
	else:
		print(f'None protocol recomended based on the user question:\n {userQuestion}')


	#TIMES
	# print(f'Time constants: {constantEND_time - start_time} s')
	# print(f'Time parseUserQuestion_time: {parseUserQuestion_time - constantEND_time} s')
	# print(f'Time embedUserQuestion_time: {embedUserQuestion_time - parseUserQuestion_time} s')
	# print(f'Time searchOnIndexFaiss_time: {searchOnIndexFaiss_time - embedUserQuestion_time} s')
	# print(f'Time evaluateCorrelations_time: {evaluateCorrelations_time - searchOnIndexFaiss_time} s')
	# print(f'Time findProtocolsRecomended_time: {findProtocolsRecomended_time - evaluateCorrelations_time} s')
	print(f'Time TOTAL: {findProtocolsRecomended_time - start_time} s')