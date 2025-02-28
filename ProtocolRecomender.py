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

######CONSTANTS
SIZE_USER_QUESTION = 100
INDEX_VECTOR_DIMENSION = 768
NPY_FILE = 'indexMap.npy'

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
    return userQuestionVector

def addVectorsToIndexFaiss():
    dictIndex = faiss.IndexFlatIP(INDEX_VECTOR_DIMENSION)
    vectorMap = np.load(NPY_FILE)

    if vectorMap.dtype != np.float32:
        vectorMap = vectorMap.astype(np.float32)

    dictIndex.add(vectorMap)

if __name__ == "__main__":
    userQuestion = parseUserQuestion()
    userQuestionVector = embedUserQuestion(userQuestion)
    addVectorsToIndexFaiss()
    searchOnIndexFaiss()
    EvaluateCorrelations()
    findProtocolsRecomended()
    sortProtocolsRecomended()
    collectReportAboutProtocol()
    printRecomendations()