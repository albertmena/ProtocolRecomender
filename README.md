# Main

This script takes the user question, embeds it, searches in the index map, and provides a list of protocols.  
An index map and a JSON file with the map for the index are required. (The `indexGenerator` script generates the index and the JSON.)

## Requirements

### DeepSeek
- [DeepSeek Model](https://ollama.com/library/deepseek-r1:14b)
- Install `ollama` andpull the model:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh 
  ollama pull deepseek-r1:14b
  ```
  model = "deepseek-r1:14b"


### Embedding
  HuggingFaceEmbeddings
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source $HOME/.cargo/env
  pip install -qU langchain-huggingface
  ```
  model_name="sentence-transformers/all-mpnet-base-v2"
  

### Indexing
  faiss-cpu
  numpy
  


#  indexGenerator

A Scipion enviroment has to be created. The script can install all the plugins of Scipion, takes all the protocol, embedd all of them and save it in an index map (numpy array) and a json file with the plugin-protocol-index references.

## Preparation
An Scipion enviroment with Scipion installed  python3 -m scipioninstaller -conda -n scipionProtocolRecomender -noAsk scipionProtocolRecomender
1. In terminal, activate the enviroment
2. Goes to the path the Scipion is installed
3. ```bash
    python3 indexGenerator.py
    ```
4. If INSTALL_PLUGINS is True will install all the plugins

## Requirements
- os
- pathlib
- requests
- subprocess
- json
- numpy
- ollama
  - model 43GB:
    ```bash
    ollama pull deepseek-r1:70b
    ```
    model = "deepseek-r1:70b"
  - model 404GB
    ```bash
    ollama pull deepseek-r1:671b
    ```
    model = "deepseek-r1:671b"
  
