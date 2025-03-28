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
from pathlib import Path
from datetime import datetime

#####CONFIGURATIONS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#'sentence-transformers/gtr-t5-large'
modelDS = "deepseek-r1:70b" #70b 43GB
modelMistral = "mistral" #7b 4.1GB
modelWizardCode = "wizardcoder:33b" #33b 19GB
modelLlama2 = "llama2:70b" #70b 39GB
modelCodeLlama = "codellama:70b" #70b 39GB
modelDSv2 = "deepseek-v2:236b" #236b 133GB -> problemas de memoria
modelLlama33 = "llama3.3" #70b 43GB

now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
fileName = f'Map_{formatted_date}'
PATH_MAP = os.path.join(Path.cwd(), fileName)

INSTALL_PLUGINS = False
SCIPION_ENVIROMENT_NAME = "scipionProtocolRecomender"
PATH_SCIPION_INSTALLED = '/home/agarcia/scipionProtocolRecomender'
SITE_PACKAGES ='/home/agarcia/miniconda/envs/scipionProtocolRecomender/lib/python3.8/site-packages'

#####CONSTANTS
SUMMARY = 'summary'
SCIENCE = 'science'
HELP = 'help'
splitter = '------'
fileDS = 'protocolsDescriptions.txt'
NPY_FILE= 'indexMap.npy'
FAISS_FILE= 'indexMap.faiss'
JSON_MAP = 'indexMap.json'
INDEX_VECTOR_DIMENSION = 768

questionSummary = [f'Can you provide a summary on no more than 100 words to a basic user on what the protocol "', '" of the plugin"', '" of Scipion does? Scipion is a software of cryoem. Please avoid any corrections or enhancements:\n']
questionParameters = [f'Summarize the purpose of each label in the _defineParameters function of the tool "', '" an open software project. Be concise and report only 10 words for each label:\n']
questionScience =  ['Provide only a summary of 100 words of what the protocol "', '" of the plugin "', '" of Scipion does. This is the main summary: ', '\nThis are the definitions of the parameters of the protocol:\n']

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
        subdirs = [Path(protocolFiles, d) for d in os.listdir(protocolFiles) if (protocolFiles / d).is_dir() and "__pycache__" not in d]
        listFolders = [protocolFiles]
        if subdirs:
            listFolders.extend(subdirs)
        for path in listFolders:
            for file in path.iterdir():
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


def responseDeep(question: str, modelI:str):
    response: ChatResponse = chat(
        model=modelI, messages=[
            {'role': 'user',
            'content': question,
            }
        ],stream=False
    )
    resp = response.message.content
    if modelI == modelDS:
        resp = resp[resp.find('</think>') + 8:]
    if modelI == modelCodeLlama:
        resp = resp.replace('**', '')
    if modelI == modelLlama33:
        indexA = resp.find('Here is a 100-word summary')
        if indexA != -1:
            indexB = resp[indexA:].find('\n')
            resp = resp[:indexA] + resp[indexB:]

        resp = resp.replace('**', '')
    return resp

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

def defineParamsTexted(scriptTexted):
    stringFunc = ''
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in ['_defineProcessParams', '_defineParams', '_defineImportParams', '_defineAcquisitionParams']:
                start_line = node.lineno - 1  # Las líneas en AST comienzan desde 1
                end_line = getattr(node, "end_lineno", None)
                if end_line is None:
                    end_line = max((child.lineno for child in ast.walk(node) if hasattr(child, "lineno")), default=start_line)
                stringFunc += "\n".join(scriptTexted.splitlines()[start_line:end_line]) + "\n\n"
    return stringFunc


def helpProtocolStr(scriptTexted):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.body and isinstance(node.body[0], ast.Expr):
                docstring_node = node.body[0].value
                if isinstance(docstring_node, ast.Str) or isinstance(docstring_node, ast.Constant):
                    return docstring_node.s
    return None


def extract_label_protocol(scriptTexted):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):  #
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == '_label':
                            if isinstance(stmt.value, ast.Str) or isinstance(stmt.value, ast.Constant):
                                return stmt.value.s


def embedPhrases(listPhrases):
    listVectors = []
    for p in listPhrases:
        listVectors.append(embeddings.embed_query(p))
    return listVectors

def removeJumpLine(string):
    return string.replace('\n', ' ')

def requestDSFillMap(dictProtocolFile):
    dictVectors = {}
    num_PLugins = len(dictProtocolFile.keys())
    index = 0
    with open(fileDS, 'w', encoding="utf-8") as fDS:
        fDS.write(f'\nDate: {date.today()}\n')
    for key in dictProtocolFile.keys():
        index+=1
        num_Protocols = len(dictProtocolFile[key])
        indexProt = 0
        with open(fileDS, 'a', encoding="utf-8") as fDS:
            fDS.write(f'\n\n###########################\nPLUGIN: {key} \n')
        print(f'\n\n#############plugin: {key}')
        print(f'Plugin {index}/{num_PLugins}')
        dictVectors[key] = {}
        for protocol in dictProtocolFile[key]:
            indexProt += 1
            print(f'\n-----------protocol {indexProt}/{num_Protocols}: {protocol} ')
            dictVectors[key][protocol] = {}
            file = dictProtocolFile[key][protocol]
            with open(file, "r", encoding="utf-8") as f:
                scriptTexted = f.read()
                protocolString = classTexted(scriptTexted, protocol)
                helpProtocol = removeJumpLine(helpProtocolStr(protocolString))
                labelProtocol = extract_label_protocol(protocolString)
                defineParamsString = defineParamsTexted(protocolString)

                time0 = time.time()
                question = questionSummary[0] + labelProtocol + questionSummary[1] +  key + questionSummary[2] + protocolString
                summaryGeneral = removeJumpLine(responseDeep(question=question, modelI=modelDS))
                time1 = time.time()
                questionP = questionParameters[0] + labelProtocol + questionParameters[1] + defineParamsString
                paramsSummary = responseDeep(question=questionP, modelI=modelCodeLlama)
                time2 = time.time()
                questionS = questionScience[0] + labelProtocol + questionScience[1] + key + questionScience[2]  + summaryGeneral + questionScience[3] + paramsSummary
                scienceSummary = responseDeep(question=questionS, modelI=modelLlama33)
                time3 = time.time()

                dictVectors[key][protocol][HELP] = embedPhrases(helpProtocol.split('.'))
                dictVectors[key][protocol][SUMMARY] = embedPhrases(summaryGeneral.split('.'))
                dictVectors[key][protocol][SCIENCE] = embedPhrases(scienceSummary.split('.'))
                print(f'{len(dictVectors[key][protocol][HELP])} vectors HELP\n{len(dictVectors[key][protocol][SUMMARY])} vectors SUMMARY\n{len(dictVectors[key][protocol][SCIENCE])} vectors PARAMETERS')
                print(f'Time summary request:  {(time1 - time0)/60} min')
                print(f'Time parameters request:  {(time2 - time1)/60} min')
                print(f'Time science request:  {(time3 - time2)/60} min')

                with open (fileDS, 'a', encoding="utf-8") as fDS:
                    fDS.write(f'-------------------------------------\nPLUGIN: {key}\nPROTOCOL: {protocol}\nHELP: {helpProtocol}\nPARAMS:{paramsSummary}\nSUMMARY: {summaryGeneral}\nSCIENCE: {scienceSummary}\n\n')
    return dictVectors

def savingDictListVect2(dictIndexMap, plugin, protocol, rowCounter):
    for b in [HELP, SUMMARY, SCIENCE]:
        for item in list(range(len(dictVectors[plugin][protocol][b]))):
            stepIndex = item + 1 #loop starts with 0, we need 1 to increase the value
            dictIndexMap["VECTORS"][rowCounter + stepIndex] = {'PLUGIN':plugin, 'PROTOCOL':protocol, 'BLOC': b}
        rowCounter += stepIndex
    return rowCounter

def indexMap(dictVectors):
    indexMapArray = np.empty((0, 768))
    dictIndexMap = {'header':{"description": "This JSON file contains sentence embeddings.",
                              "index_info": "Each value represent the index in the indexMap.npy that represent the embeddig of each phrase.",
                              "usage": "These embeddings can be easy search with the plugin, protocol and summary.",
                              "Date": f'{date.today()}',
                              "Plugins collected": ', '.join(list(dictProtocolFile.keys()))},
                    "VECTORS": {},
                    }
    rowCounter = -1
    for plugin in dictVectors.keys():
        for protocol in dictVectors[plugin]:
            arrayHelp = np.array(dictVectors[plugin][protocol][HELP])
            arraySummary = np.array(dictVectors[plugin][protocol][SUMMARY])
            arrayScience = np.array(dictVectors[plugin][protocol][SCIENCE])
            indexMapArray = np.vstack([indexMapArray, arrayHelp])
            indexMapArray = np.vstack([indexMapArray, arraySummary])
            indexMapArray = np.vstack([indexMapArray, arrayScience])
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
    os.makedirs(PATH_MAP, exist_ok=True)
    os.chdir(PATH_MAP)
    listOfPlugins, dictPlugins = listPlugins()
    if INSTALL_PLUGINS: installAllPlugins(listOfPlugins, dictPlugins)
    dictProtocolFile = readingProtocols()
    #dictProtocolFile = { 'aretomo': dictProtocolFile['aretomo']} #JUST TO DEBUG
    #dictProtocolFile = {'motioncorr': dictProtocolFile['motioncorr']} #JUST TO DEBUG
    dictProtocolFile = { 'xmipp3': dictProtocolFile['xmipp3']} #JUST TO DEBUG
    dictVectors = requestDSFillMap(dictProtocolFile)
    indexMap(dictVectors)
    writtingIndexFaissFile()