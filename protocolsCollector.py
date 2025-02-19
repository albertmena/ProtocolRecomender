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

'''Create a scipion installation with an Enviroment named:
python3 -m scipioninstaller -conda -n scipionProtocolRecomender -noAsk scipionProtocolRecomender
In terminal, activate the enviroment
conda activatescipionProtocolRecomender
Goes to the path the Scipion is installed
run: python3 protocolsCollector.py
If INSTALL_PLUGINS is True will install all the plugins
'''
import os

INSTALL_PLUGINS = False
SCIPION_ENVIROMENT_NAME = "scipionProtocolRecomender"
PATH_SCIPION_INSTALLED = '/home/agarcia/develops/scipionProtocolRecomender'
SITE_PACKAGES = '/home/agarcia/miniconda/envs/scipionProtocolRecomender/lib/python3.8/site-packages/'
listOfPlugins = []
pluginsNoInstalled = []
dictPlugins = {}
#### List of plugins
import requests

urlAllPluginsRegistered = "https://scipion.i2pc.es/getplugins/"
response = requests.get(urlAllPluginsRegistered)

if response.status_code == 200:
    data = response.json()
    for key, value in data.items():
        listOfPlugins.append(value["pipName"])
        dictPlugins[key] = value["name"]
else:
    print(f"Error {response.status_code}: No se pudo obtener el JSON")

#### Install plugins without binaries
import subprocess

# Nombre del entorno y los paquetes a instalar
def installAllPlugins():
    for plugin in dictPlugins.keys():
        activate_env = f"./scipion3 installp -p  {plugin} --noBin"
        try:
            result = subprocess.run(activate_env, shell=True, check=True, cwd=PATH_SCIPION_INSTALLED)
            if result.returncode == 0:
                print(f"{plugin}: Installed\n\n")
            else:
                print(f"{plugin} Error: {result}\n\n")
        except Exception as e:
            print(f'Error: {e}')
            pluginsNoInstalled.append(plugin)
        subprocess.run(f'./scipion3 pip list', shell=True, check=True, cwd=PATH_SCIPION_INSTALLED)

    pluginsInstalled = [p for p in listOfPlugins if p not in pluginsNoInstalled]
    print(f'Plugins installed ({len(pluginsInstalled)}): {pluginsInstalled}\nPlugins no installed ({len(pluginsNoInstalled)}): {pluginsNoInstalled}')

if INSTALL_PLUGINS: installAllPlugins()

#### List protocols for each plugin
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


### ASK DEEPSEEK DOR DESCRIPTION
from ollama import chat
from ollama import ChatResponse

questionForProtocols= 'Describe everything this Scipion protocol does. First, provide a summary (200 words) with the main keywords. Then, explain what does all the parameter (defineParameters) (200 words). Finally, describe the inputs and outputs (200 words). Omit any tittle in the three steps: \n'
splittersSummary1 = 'defineParameters'
splittersSummary2 = 'Inputs and Outputs'

def responseDeepSeek(ProtocolQuestion:str ):
    response: ChatResponse = chat(
        model=model, messages=[
            {'role': 'user',
            'content': questionForProtocols + ProtocolQuestion,
            }
        ],stream=False
    )
    resp = response.message.content
    print(resp)
    summary = resp[resp.find('</think>') + 8:]
    return summary

model = "deepseek-r1:14b" #9GB 14.8B parameters

def protocol2Text(pathProtocol):
    with open(pathProtocol, 'r') as archivo:
        return archivo.read()

for plugin in protocol_dict.keys():
    for protocol in protocol_dict[plugin]:
        responseDeepSeek(protocol2Text(os.path.join(SITE_PACKAGES, plugin, 'protocols', protocol)))



#### Ask deepseek about description of each protocol


#### Split description


####Split phrases


##### EMmbdding phrases


##### Saving vectors
