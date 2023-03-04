from typing import Union, List, Mapping, Optional
import enum
import logging
from pathlib import Path
import getpass

import openai
import tiktoken
import pandas as pd
import numpy as np

from openai.embeddings_utils import distances_from_embeddings

class ChatBot:

    class Language(enum.Enum):
        EN = 'en',
        IT = 'it',

    PROMPT_TEMPLATES = {
        'en':
            "{identity} "
            "Answer the question based on the context below, and if the question can't be "
            "answered based on the context, say \"I don't know.\"\n\n"
            "Context: {context}\n\n",
        'it':
            "{identity} "
            "Rispondi alla domanda con in base al contesto sottostante e, se non è possibile "
            "rispondere alla domanda in base al contesto, dì \"Non lo so.\"\n\n"
            "Contesto: {context}\n\n"
    }

    def __init__(self,
        kb_name: str,
        openai_api_key: str,
        tokenizer_encoding: str = 'cl100k_base',
        embeddings_engine: str = 'text-embedding-ada-002',
        model_name: str = 'gpt-3.5-turbo',
        max_prompt_len: int = 1800,
        max_kb_article_len: int = 50,
        max_response_len: int = 200,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        openai.api_key = openai_api_key
        self.kb_name = kb_name
        self.data_folder = Path('data') / kb_name
        self.kb_path = self.data_folder / Path('qa.csv')
        self.identity_path = self.data_folder / Path('identity.txt')
        self.embeddings_path = self.data_folder / Path('embeddings.csv')
        self.kb_lang_path = self.data_folder / Path('lang.txt')
        with open(self.kb_lang_path) as f:
            self.kb_language = f.read().strip()
        if self.kb_language not in self.PROMPT_TEMPLATES:
            raise ValueError(
                f'Unsupported knowledge base langue "{self.kb_language}". '
                f'Supported languages: {", ".join(self.PROMPT_TEMPLATES.keys())}'
            )
        self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        self.embeddings_engine_name = embeddings_engine
        self.model_name = model_name
        self.max_prompt_len = max_prompt_len
        self.max_kb_article_len = max_kb_article_len
        self.max_response_len = max_response_len
        self.history: List[Mapping[str, str]] = []
        with open(self.identity_path) as f:
            self.identity = f.read().strip()
        if self.embeddings_path.is_file():
            self.logger.info('Embeddings file found, trying to load it ...')
            df = pd.read_csv(self.embeddings_path)
            df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
            self.embeddings = df
            self.logger.info('Embeddings were successfully loaded.')
        else:
            self.logger.info(
                'Embeddings file not found, creating embeddings with OpenAI services. '
                'This might take several minutes based on the knowledge base size ...'
            )
            # lines = self._shortened(self.kb_path, max_tokens=self.max_kb_article_len)
            self.embeddings = self._create_embeddings(
                output_path=self.embeddings_path
            )
            self.logger.info('Embeddings were successfully created.')

    def _split_into_many(self, text: str, max_tokens: int = 500) -> List[str]:
        ''' Function to split the text into chunks of a maximum number of tokens '''

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(self.tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is
            # greater than the max number of tokens, then add the chunk to the list of chunks and
            # reset the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    def _shortened(self, kb_path, max_tokens: int = 500) -> List[str]:
        ''' Ensures each kb article is shorter than max_tokens number of tokens. '''
        shortened = []
        with open(kb_path) as f:
            for line in f:
                line = line.strip()
                # If the text is None, go to the next row
                if line is None:
                    continue

                # If the number of tokens is greater than the max number of tokens, split the text
                # into chunks
                if len(self.tokenizer.encode(line)) > max_tokens:
                    shortened += self._split_into_many(line, max_tokens=max_tokens)

                # Otherwise, add the text to the list of shortened texts
                else:
                    shortened.append(line)
        return shortened

    def _create_embeddings(self,
        output_path: Union[str, Path],
        # lines: Sequence[str]
    ) -> pd.DataFrame:
        ''' Create embeddings for kb articles using OpenAi services. '''
        df = pd.read_csv(self.kb_path)
        df['embeddings'] = df['question'].apply(
            lambda x: openai.Embedding.create(
                input=x,
                engine=self.embeddings_engine_name
            )['data'][0]['embedding']
        )
        df['n_tokens'] = df['answer'].apply(
            lambda x: len(self.tokenizer.encode(" " + x))
        )
        df.to_csv(output_path, index=False)
        return df

    def create_context(self, question: str) -> str:
        """
        Create a context for a question by finding the most similar context from the embeddings
        """

        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(
            input=question,
            engine=self.embeddings_engine_name
        )['data'][0]['embedding']

        df = self.embeddings
        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(
            q_embeddings,
            df['embeddings'].values,
            distance_metric='cosine'
        )

        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for _, row in df.sort_values('distances', ascending=True).iterrows():

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > self.max_prompt_len:
                break

            # Else add it to the text that is being returned
            returns.append(row["answer"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def answer_question(self,
        question: str,
        stop_sequence: Optional[str] = None,
        temperature: float = 0.0,
        debug=False
    ) -> str:
        """Answer a question based on the most similar context from the kb."""
        context = self.create_context(question)
        self.logger.debug("Context:\n%s", context)
        if debug:
            print("Context:\n{}".format(context))
            print("\n--------\n")
        prompt_template = self.PROMPT_TEMPLATES[self.kb_language]
        prompt = prompt_template.format(
            context=context,
            question=question,
            identity=self.identity
        )

        messages = [
            {"role": "system", "content": prompt}
        ]
        messages += self.history
        messages += [{"role": "user", "content": question}]

        try:
            # Create a completions using the question and context
            api_resp = openai.ChatCompletion.create(
                messages=messages,
                temperature=temperature,
                stop=stop_sequence,
                max_tokens=self.max_response_len,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model=self.model_name,
            )
            response = api_resp["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            self.logger.exception(e)
            return ""


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Virtual assistant chatbot")
    parser.add_argument('-kb', '--knowledge_base', type=str, required=True,
                        help="knowledge base (./data subfolder)")
    args = parser.parse_args()

    openai_api_key = getpass.getpass('Enter OpenAI API key: ')

    agent = ChatBot(
        kb_name=args.knowledge_base,
        openai_api_key=openai_api_key,
        max_prompt_len=400,
        max_response_len=400
    )

    print(
        f'\nWelcome to {agent.kb_name} virtual assistant. Type "exit" to exit. '
        f'Start by typing a question in "{agent.kb_language}" language.'
    )
    while True:
        question = input('\nUser: ')
        if question.lower() == 'exit':
            break
        answer = agent.answer_question(question)
        print('\nAssistant:', answer)
