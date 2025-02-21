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

#####CONSTANTS
SUMMARY = 'summary'
PARAMETERS = 'parameters'
IO = 'IO'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
model = "deepseek-r1:14b" #9GB 14.8B parameters
INSTALL_PLUGINS = False
SCIPION_ENVIROMENT_NAME = "scipionProtocolRecomender"
PATH_SCIPION_INSTALLED = '/home/agarcia/develops/scipionProtocolRecomender'
SITE_PACKAGES = '/home/agarcia/miniconda/envs/scipionProtocolRecomender/lib/python3.8/site-packages/'
listOfPlugins = []
pluginsNoInstalled = []
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

#### INSTALL PLUGINS WITHOUT BINARIES

def runInstallPlugin(plugin, noBin:bool=True):
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

def installAllPlugins():
    dictPlugins.pop('scipion-em-xmipp2', None)
    for plugin in dictPlugins.keys():
        runInstallPlugin(plugin)
        subprocess.run(f'./scipion3 pip list', shell=True, check=True, cwd=PATH_SCIPION_INSTALLED)

    pluginsInstalled = [p for p in listOfPlugins if p not in pluginsNoInstalled]
    print(f'Plugins installed ({len(pluginsInstalled)}): {pluginsInstalled}\nPlugins no installed ({len(pluginsNoInstalled)}): {pluginsNoInstalled}')

if INSTALL_PLUGINS: installAllPlugins()

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


protocol_dict["chimera"] = protocol_dict.pop("chimerax")

blackList = ['pyworkflowtests', 'xmipp2']
for p in blackList:
    protocol_dict.pop(p, None)

dictProtocolFile = {}
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


print(f'Registred: {len(dictProtocolFile)} plugins')
totalProtocols = 0
for key in dictProtocolFile.keys():
    numProts = len(dictProtocolFile[key])
    totalProtocols+= numProts
    print(f'{key}: {len(dictProtocolFile[key])} protocols')

print(f'Total protocols registred: {totalProtocols}')



### ASK DEEPSEEK DOR DESCRIPTION

questionForProtocols= 'Describe everything this Scipion protocol does in three blocs divided each one by a string like this ------. First, provide a summary (200 words) with the main keywords. Then, explain what does all the parameter (defineParameters) (200 words). Finally, describe the inputs and outputs (200 words). Omit any tittle in the three blocs: \n'
splittersSummary1 = 'defineParameters'
splittersSummary2 = 'Inputs and Outputs'

def responseDeepSeek(ProtocolStr:str ):
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


def splitPhrasesDescription(stringFull):
    summaryPhrases = stringFull[:stringFull.find(splittersSummary1)].split('.')
    parametersPhrases = stringFull[stringFull.find(splittersSummary1):stringFull.find(splittersSummary2)].split('.')
    IOPhrases = stringFull[stringFull.find(splittersSummary2):].split('.')

    return summaryPhrases, parametersPhrases, IOPhrases


def protocol2Text(pathProtocol):
    with open(pathProtocol, 'r') as archivo:
        return archivo.read()


def classTexted(scriptTexted):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in protocol:
                start_line = node.lineno - 1  # ast usa 1-based indexing
                end_line = node.end_lineno if hasattr(node,"end_lineno") else None
                if end_line is None:
                    for child_node in ast.walk(node):
                        if hasattr(child_node,
                                   "lineno") and child_node.lineno > start_line:
                            end_line = child_node.lineno
                return "\n".join(
                    scriptTexted.splitlines()[start_line:end_line])


def embedPhrases(listPhrases):
    listVectors = []
    for p in listPhrases:
        listVectors.append(embeddings.embed_query(p))
    return listVectors


dictVectors = {}
for key in dictProtocolFile.keys():
    dictVectors[key] = {}
    for protocol in dictProtocolFile[key]:
        dictVectors[key][protocol] = {}
        file = dictProtocolFile[key][protocol]
        with open(file, "r", encoding="utf-8") as f:
            scriptTexted = f.read()
            protocolString = classTexted(scriptTexted)
            descriptionProtocol = responseDeepSeek(ProtocolStr=protocolString)
            summaryPhrases, parametersPhrases, IOPhrases = splitPhrasesDescription(descriptionProtocol)
            dictVectors[key][protocol][SUMMARY] = embedPhrases(summaryPhrases)
            dictVectors[key][protocol][PARAMETERS] = embedPhrases(parametersPhrases)
            dictVectors[key][protocol][IO] = embedPhrases(IOPhrases)



#### Split description


####Split phrases


##### EMmbdding phrases


##### Saving vectors
indexMapArray = np.array()
dictIndexMap = {'header':{"description": "This JSON file contains sentence embeddings.",
                          "index_info": "Each value represent the index in the indexMap.npy that represent the embeddig of each phrase.",
                          "usage": "These embeddings can be easy search with the plugin, protocol and bloc (summary, parameters, IO).",
                          "Date": f'{date.today()}\n',
                          "Plugins collected": f'{dictProtocolFile.keys()}'
                },
                "DATA": None,
                }

def savingDictListVect2(dictIndexMap, plugin, protocol, rowCounter):
    for b in [SUMMARY, PARAMETERS, IO]:
        for item in list(range(dictVectors[key][protocol][b])):
            stepIndex = item + 1 #loop starts with 0, we need 1 to increase the value
            dictIndexMap["DATA"][rowCounter + stepIndex] = f'PLUGIN: {plugin} PROTOCOL: {protocol} BLOC: {b}'
        rowCounter += stepIndex
    return rowCounter


for plugin in dictVectors.keys():
    rowCounter = -1
    for protocol in dictVectors[plugin]:
        indexMapArray = np.vstack([indexMapArray, dictVectors[key][protocol][SUMMARY]])
        indexMapArray = np.vstack([indexMapArray, dictVectors[key][protocol][PARAMETERS]])
        indexMapArray = np.vstack([indexMapArray, dictVectors[key][protocol][IO]])
        rowCounter = savingDictListVect2(dictIndexMap, plugin, protocol, rowCounter)


np.save('indexMap.npy', indexMapArray)
with open("indexMap.json", "w", encoding="utf-8") as f:
    json.dump(dictIndexMap, f, indent=4, ensure_ascii=False)






#list plugins
'''Plugins installed (61): ['scipion-em-eman2', 'scipion-em-cryoassess', 
'scipion-em-resmap', 'scipion-em-deepfinder', 'scipion-em-relion', 
'scipion-em-cryosparc2', 'scipion-em-cryosparc2', 'scipion-em-imod', 

'scipion-em-embuild', 'scipion-em-pyseg', 'scipion-em-repic', 'scipion-em-flexutils',
'scipion-em-kiharalab', 'scipion-em-spoc', 'scipion-em-fsc3d', 'scipion-em-locscale',
'scipion-em-cistem', 'scipion-em-tomo3d', 'scipion-em-roodmus', 'scipion-em-fidder', 
'scipion-em-imagic', 'scipion-em-atomstructutils', 'scipion-em-cryodrgn', 
'scipion-em-motioncorr', 'scipion-em-facilities', 'scipion-em-warp', 'scipion-em-gctf',
'scipion-em-phenix', 'scipion-em-atsas', 'scipion-em-spider', 'scipion-em-aretomo',
'scipion-em-sidesplitter', 'scipion-em-ccp4', 'scipion-em-empiar', 'scipion-em-chimera',
'scipion-em-tomoviz', 'scipion-em-appion', 'scipion-em-gautomatch', 'scipion-em-emready',
'scipion-em-sphire', 'scipion-em-xmipp2', 'scipion-em-tomosegmemtv', 'scipion-em-continuousflex',
'scipion-em-cryoef', 'scipion-em-reliontomo', 'scipion-em-isonet', 'scipion-em-topaz',
'scipion-em-bsoft', 'scipion-em-emantomo', 'scipion-em-emxlib', 'scipion-em-dynamo', 
'scipion-em-localrec', 'scipion-em-modelangelo', 'scipion-em-tomo', 'scipion-em-susantomo',
'scipion-em-xmipp', 'scipion-em-xmipptomo', 'scipion-em-tomotwin', 'scipion-em-bamfordlab',
'scipion-em-novactf', 'scipion-em-prody']
'''