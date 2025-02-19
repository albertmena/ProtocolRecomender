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
INSTALL_PLUGINS = True
SCIPION_ENVIROMENT_NAME = "scipionProtocolRecomender"
PATH_SCIPION_INSTALLED = '/home/agarcia/develops/scipionProtocolRecomender'
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
for plugin in dictPlugins.keys():
    pass

#### Ask deepseek about description of each protocol


#### Split description


####Split phrases


##### EMmbdding phrases


##### Saving vectors
