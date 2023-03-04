# Virtual assistant chatbot proof-of-concept

This is a proof-of-concept of a virtual assistant that answers questions based on a predefined knowledge base.

## Installation

Install the python dependencies in your virtual environment from requirements.txt:

```shell
pip install -r requirements.txt
```

## Usage - CLI

You can try the chatbot with the command line interface provided:

```text
usage: chatbot.py [-h] -kb KNOWLEDGE_BASE

Virtual assistant chatbot

options:
  -h, --help            show this help message and exit
  -kb KNOWLEDGE_BASE, --knowledge_base KNOWLEDGE_BASE
                        knowledge base (./data subfolder)
```

The knowledge base parameter should be the name of one of the subdirectory of the `data` directory.

Have ready your OpenAI API key, the CLI will ask for it.

## Usage - Streamlit

There is also a streamlit interface so you can interact with the agent from a web window. First, create a subdirectory and a file in it called `.streamlit/secrets.toml`. The contents of the file should be:

```text
api_secret = "[your OpenAI API key]"
```

Now you can launch the streamlit app with:

```bash
streamlit run chatlit.py
```

If it starts, open http://localhost:8501 to interact with the agent.

## Adding a new knowledge base

1. Create a subdirectory of the `data` directory with the name of your knowledge base. We will refer to this directory as "working directory" in the following points.
2. Create a text file named `lang.txt` in the working directory, and place the language code of the knowledge base in it. Currently "it" (Italian) and "en" (English) are supported.
3. Create a text file named `identity.txt` in the working directory. Shortly describe the its identity to the bot in the language of the knowledge base.
4. Create a comma-separated-value (CSV) file in the working directory, called `qa.csv`. The table in this file should contain two columns, "question" and "answer". Write the FAQ or Q&A of your knowledge base in this file. The question column can be a question, a short summary of the argument, or a list of keywords. It should be as much as relevant to the topic as possible The answer should discuss the argument in question in maximum 500 words.

The first time you start the bot with a new knowledge base, the embeddings of the kb will be created. This can take up to several minutes, be patient.
