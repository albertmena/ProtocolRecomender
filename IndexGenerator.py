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

import os
import requests
import subprocess
from ollama import chat
from ollama import ChatResponse
import ast
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import json
from datetime import date
import time
import faiss #faiss-cpu


#####CONFIGURATIONS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#'sentence-transformers/gtr-t5-large'
model = "deepseek-r1:70b" #40GC 70.6B parameters
INSTALL_PLUGINS = False
SCIPION_ENVIROMENT_NAME = "scipionProtocolRecomender"
PATH_SCIPION_INSTALLED = '/home/agarcia/scipionProtocolRecomender'
SITE_PACKAGES ='/home/agarcia/miniconda/envs/scipionProtocolRecomender/lib/python3.8/site-packages'

#####CONSTANTS
SUMMARY = 'summary'
PARAMETERS = 'parameters'
splitter = '------'
fileDS = 'protocolsDescriptions.txt'
NPY_FILE= 'indexMap.npy'
FAISS_FILE= 'indexMap.faiss'
JSON_MAP = 'indexMap.json'
INDEX_VECTOR_DIMENSION = 768

#questionForProtocols= f'Describe everything this Scipion protocol does in two blocs divided by {splitter} First, provide a summary (200 words) with the main keywords. Then, explain what does all the parameter (defineParameters) (200 words). Omit any tittle in the two blocs: \n'
questionSummary = f'Can you provide a concise summary of around 150 words of this Scipion protocol? Please avoid any corrections, commentary or enhancements. Stick strictly to the original content.:\n'
#questionParameters = 'Can you provide a concise summary of around 200 words of the parameters of this Scipion protocol?'
questionParameters = "Please provide a concise description, of the parameters defined in the defineParameters section of this Scipion protocol. Report just around 10 words for each parameter. Focus solely on listing and briefly explaining each parameter without making corrections or adding extra commentary. Stick strictly to the original content:\n"

def listPlugins():
    listOfPlugins = []
    dictPlugins = {}

    #### LIST OF PLUGINS
    urlAllPluginsRegistered = "https://scipion.i2pc.es/getplugins/"
    response = requests.get(urlAllPluginsRegistered)

    if response.status_code == 200:
        data = response.json()
        for key, value in data.items():
            listOfPlugins.append(value["pipName"])
            dictPlugins[key] = value["name"]
    else:
        print(f"Error {response.status_code}: No se pudo obtener el JSON")

    # dictPlugins['pkpd'] = 'pkpd'
    # dictPlugins['chem'] = 'pwchem'
    # listOfPlugins.append('scipion-pkpd')
    # listOfPlugins.append('scipion-chem')
    return listOfPlugins, dictPlugins

def runInstallPlugin(plugin, pluginsNoInstalled, noBin:bool=True):
    if noBin: flagBin = '--noBin'
    else: flagBin=''
    activate_env = f"./scipion3 installp -p  {plugin} {flagBin}"
    try:
        result = subprocess.run(activate_env, shell=True, check=True,
                                cwd=PATH_SCIPION_INSTALLED)
        if result.returncode == 0:
            print(f"{plugin}: Installed\n\n")
        else:
            print(f"{plugin} Error: {result}\n\n")
    except Exception as e:
        print(f'Error: {e}')
        pluginsNoInstalled.append(plugin)

def installAllPlugins(listOfPlugins, dictPlugins):
    pluginsNoInstalled = []
    dictPlugins.pop('scipion-em-xmipp2', None)
    for plugin in dictPlugins.keys():
        runInstallPlugin(plugin, pluginsNoInstalled)
        subprocess.run(f'./scipion3 pip list', shell=True, check=True, cwd=PATH_SCIPION_INSTALLED)

    pluginsInstalled = [p for p in listOfPlugins if p not in pluginsNoInstalled]
    print(f'Plugins installed ({len(pluginsInstalled)}): {pluginsInstalled}\nPlugins no installed ({len(pluginsNoInstalled)}): {pluginsNoInstalled}')

def readingProtocols():
    #### LIST PROTOCOLS
    protocol_dict = {}
    result = subprocess.run(f'./scipion3 protocols', shell=True, check=True,cwd=PATH_SCIPION_INSTALLED, capture_output=True, text=True)
    protocolsStr = result.stdout
    protocolsStr = protocolsStr[protocolsStr.find('LABEL') + 5:]

    for line in protocolsStr.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            package = parts[0]
            protocol = parts[1]
            if package not in protocol_dict:
                protocol_dict[package] = []
            protocol_dict[package].append(protocol)

    protocol_dict.pop("Scipion", None)
    protocol_dict["chimera"] = protocol_dict.pop("chimerax")
    blackList = ['pyworkflowtests', 'xmipp2']
    for p in blackList:
        protocol_dict.pop(p, None)

    dictProtocolFile = {}
    from pathlib import Path
    for plugin in protocol_dict.keys():
        #print(f'PLUGIN: {plugin}')
        dictProtocolFile[plugin] = {}
        protocolFiles = Path(os.path.join(SITE_PACKAGES, plugin, 'protocols'))
        for file in protocolFiles.iterdir():
            if file.is_file() and file.suffix == ".py" and file.name != "__init__.py":
                with open(file, "r", encoding="utf-8") as f:
                    scriptTexted = f.read()
                    tree = ast.parse(scriptTexted)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if node.name in protocol_dict[plugin]:
                                #print(f"Clase encontrada: {node.name}")
                                dictProtocolFile[plugin].update({node.name: file})

    protocol_dict.pop("Scipion", None)
    print(f'Registred: {len(dictProtocolFile)} plugins')
    totalProtocols = 0
    for key in dictProtocolFile.keys():
        numProts = len(dictProtocolFile[key])
        totalProtocols+= numProts
        print(f'{key}: {len(dictProtocolFile[key])} protocols')

    print(f'Total protocols registred: {totalProtocols}')
    return dictProtocolFile


def responseDeepSeek(questionForProtocols: str, ProtocolStr:str ):
    response: ChatResponse = chat(
        model=model, messages=[
            {'role': 'user',
            'content': questionForProtocols + ProtocolStr,
            }
        ],stream=False
    )
    resp = response.message.content
    summary = resp[resp.find('</think>') + 8:]
    return summary

def protocol2Text(pathProtocol):
    with open(pathProtocol, 'r') as archivo:
        return archivo.read()

def classTexted(scriptTexted, protocol):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in protocol:
                start_line = node.lineno - 1  # ast use 1-based indexing
                end_line = node.end_lineno if hasattr(node,"end_lineno") else None
                if end_line is None:
                    for child_node in ast.walk(node):
                        if hasattr(child_node,
                                   "lineno") and child_node.lineno > start_line:
                            end_line = child_node.lineno
                return "\n".join(
                    scriptTexted.splitlines()[start_line:end_line])
#
def embedPhrases(listPhrases):
    listVectors = []
    for p in listPhrases:
        listVectors.append(embeddings.embed_query(p))
    return listVectors


def requestDSFillMap(dictProtocolFile):
    dictVectors = {}
    num_PLugins = len(dictProtocolFile.keys())
    index = 0
    with open(fileDS, 'w', encoding="utf-8") as fDS:
        fDS.write(f'\nDate: {date.today()}\n')
    for key in dictProtocolFile.keys():
        index+=1
        with open(fileDS, 'a', encoding="utf-8") as fDS:
            fDS.write(f'\n\n###############\nPLUGIN: {key} \n')
        print(f'\n\n#############plugin: {key}')
        print(f'Plugin {index}/{num_PLugins}')
        dictVectors[key] = {}
        for protocol in dictProtocolFile[key]:
            print(f'\n-----------protocol: {protocol}')
            dictVectors[key][protocol] = {}
            file = dictProtocolFile[key][protocol]
            with open(file, "r", encoding="utf-8") as f:
                scriptTexted = f.read()
                protocolString = classTexted(scriptTexted, protocol)
                time0 = time.time()
                summaryPhrases = responseDeepSeek(questionForProtocols=questionSummary, ProtocolStr=protocolString)
                time1 = time.time()
                print(f'Time summary request:  {(time1 - time0)/60} min')
                parametersPhrases = responseDeepSeek(questionForProtocols=questionParameters, ProtocolStr=protocolString)
                print(f'Time parameters request:  {(time.time() - time1)/60} min')
                dictVectors[key][protocol][SUMMARY] = embedPhrases(summaryPhrases.split('.'))
                dictVectors[key][protocol][PARAMETERS] = embedPhrases(parametersPhrases.split('.'))
                print(f'{len(dictVectors[key][protocol][SUMMARY])} vectors SUMMARY\n{len(dictVectors[key][protocol][PARAMETERS])} vectors PARAMETERS')

            with open (fileDS, 'a', encoding="utf-8") as fDS:
                fDS.write(f'----------------------\nPLUGIN: {key}\nPROTOCOL: {protocol}\nSUMMARY: {summaryPhrases}\n\nPARAMETERS: {parametersPhrases}\n\n')
    return dictVectors

def savingDictListVect2(dictIndexMap, plugin, protocol, rowCounter):
    for b in [SUMMARY, PARAMETERS]:
        for item in list(range(len(dictVectors[plugin][protocol][b]))):
            stepIndex = item + 1 #loop starts with 0, we need 1 to increase the value
            dictIndexMap["VECTORS"][rowCounter + stepIndex] = {'PLUGIN':plugin, 'PROTOCOL':protocol, 'BLOC': b}
        rowCounter += stepIndex
    return rowCounter

def indexMap(dictVectors):
    indexMapArray = np.empty((0, 768))
    dictIndexMap = {'header':{"description": "This JSON file contains sentence embeddings.",
                              "index_info": "Each value represent the index in the indexMap.npy that represent the embeddig of each phrase.",
                              "usage": "These embeddings can be easy search with the plugin,  protocol  and blocs: summary, parameters.",
                              "Date": f'{date.today()}',
                              "Plugins collected": ', '.join(list(dictProtocolFile.keys()))},
                    "VECTORS": {},
                    }
    rowCounter = -1
    for plugin in dictVectors.keys():
        for protocol in dictVectors[plugin]:
            arraySummary = np.array(dictVectors[plugin][protocol][SUMMARY])
            arrayParameters = np.array(dictVectors[plugin][protocol][PARAMETERS])
            indexMapArray = np.vstack([indexMapArray, arraySummary])
            indexMapArray = np.vstack([indexMapArray, arrayParameters])
            rowCounter = savingDictListVect2(dictIndexMap, plugin, protocol, rowCounter)


    np.save(NPY_FILE, indexMapArray)
    with open(JSON_MAP, "w", encoding="utf-8") as f:
        json.dump(dictIndexMap, f, indent=4, ensure_ascii=False)

def writtingIndexFaissFile():
	dictIndex = faiss.IndexFlatIP(INDEX_VECTOR_DIMENSION)
	vectorMap = np.load(NPY_FILE)

	if vectorMap.dtype != np.float32:
		vectorMap = vectorMap.astype(np.float32)

	assert vectorMap.shape[1] == INDEX_VECTOR_DIMENSION, "Dimension mismatch!"

	dictIndex.add(vectorMap)
	faiss.write_index(dictIndex, FAISS_FILE)


if __name__ == "__main__":
    # listOfPlugins, dictPlugins = listPlugins()
    # listOfPlugins = ['scipion-em-motioncorr', 'scipion-em-aretomo']
    # dictPlugins = {dictPlugins['scipion-em-motioncorr'], dictPlugins['scipion-em-aretomo']}
    # if INSTALL_PLUGINS: installAllPlugins(listOfPlugins, dictPlugins)
    dictProtocolFile = readingProtocols()
    #dictProtocolFile = {'motioncorr': dictProtocolFile['motioncorr'], 'aretomo': dictProtocolFile['aretomo']} #JUST TO DEBUG
    dictProtocolFile = {'motioncorr': dictProtocolFile['motioncorr'], 'xmipp3': dictProtocolFile['xmipp3']} #JUST TO DEBUG
    dictVectors = requestDSFillMap(dictProtocolFile)
    indexMap(dictVectors)
    writtingIndexFaissFile()