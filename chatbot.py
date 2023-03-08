from typing import Union, List, Mapping, Optional, Sequence, Iterable, Iterator, NamedTuple
import enum
import logging
from pathlib import Path
import os
import csv

import openai
import tiktoken
import pandas as pd
import numpy as np
import numpy.typing

from openai.embeddings_utils import distances_from_embeddings

def read_file(filename: Union[str, Path]) -> str:
    with open(filename) as f:
        return f.read().strip()


class KnowledgeBase:

    @staticmethod
    def available_knowledge_bases() -> Sequence[str]:
        return next(os.walk('data'))[1]

    @staticmethod
    def read_file(filename: Union[str, Path]) -> str:
        with open(filename) as f:
            return f.read().strip()

    def __init__(self, kb_name: str) -> None:
        self.data_folder = Path('data') / kb_name
        self.kb_path = self.data_folder / Path('qa.csv')
        self.embeddings_path = self.data_folder / Path('embeddings.csv')
        self.identity = KnowledgeBase.read_file(self.data_folder / Path('identity.txt'))
        self.language = KnowledgeBase.read_file(self.data_folder / Path('lang.txt'))
        self.help_text = KnowledgeBase.read_file(self.data_folder / Path('help.txt'))


class Preprocessor:
    ''' Preprocesses a list of text to be used by the chatbot as knowledge base.

    Args:
        tokenizer_encoding (str): The encoding name of the tokenizer.
            `tiktoken.list_encoding_names()` returns the available encodings.
        embeddings_engine (str): The name of the embedding engine. Just use "text-embedding-ada-002"
            in most use cases. For more info:
            https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
        max_kb_article_len (int): The maximum length of the knowledge base article chunks, expressed
            in number of tokens.
    '''

    class Article(NamedTuple):
        ''' A preprocessed knowledge base article. '''

        text: str
        ''' The text of the article. '''

        embeddings: numpy.typing.NDArray
        ''' The calculated embeddings vector. '''

        n_tokens: int
        ''' The number of tokens in the article. '''

    def __init__(self,
        tokenizer_encoding: str = 'cl100k_base',
        embeddings_engine: str = 'text-embedding-ada-002',
        max_kb_article_len: int = 500,
    ):
        self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        self.embeddings_engine_name = embeddings_engine
        self.max_kb_article_len = max_kb_article_len

    def _split_into_many(self, text: str) -> Iterator[str]:
        ''' Function to split the text into chunks of a maximum number of tokens '''

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(self.tokenizer.encode(" " + sentence)) for sentence in sentences]

        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is
            # greater than the max number of tokens, then add the chunk to the list of chunks and
            # reset the chunk and tokens so far
            if tokens_so_far + token > self.max_kb_article_len:
                res = ". ".join(chunk) + "."
                yield res
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > self.max_kb_article_len:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

    def _shortened(self, kb_iterable: Iterable[str]) -> Iterator[str]:
        ''' Ensures each kb article is shorter than max_tokens number of tokens. '''
        for line in kb_iterable:
            line = line.strip()
            # If the text is None, go to the next row
            if line is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text
            # into chunks
            if len(self.tokenizer.encode(line)) > self.max_kb_article_len:
                yield from self._split_into_many(line)

            # Otherwise, add the text to the list of shortened texts
            else:
                yield line

    def create_embeddings(self, kb_iterable: Iterable[str]) -> Iterator[Article]:
        ''' Preprocess for knowledge base articles using OpenAi services.

        Make sure that you set openai.api_key before calling this method.

        Args:
            kb_iterable Iterable[str]: An iterable that yields articles of the source knowledge base.

        Returns:
            Iterator[Article]: An iterator over the preprocessed knowledge base articles
        '''
        for text in self._shortened(kb_iterable):
            print('.', end='', flush=True)
            yield Preprocessor.Article(
                text=text,
                embeddings=openai.Embedding.create(
                    input=text,
                    engine=self.embeddings_engine_name)['data'][0]['embedding'],
                n_tokens=len(self.tokenizer.encode(" " + text))
            )
        print()

class ChatBot:

    class Language(enum.Enum):
        EN = 'en',
        IT = 'it',

    PROMPT_TEMPLATES = {
        'en':
            "{identity}\n\n"
            "Answer the question as truthfully as possible using the provided context, and if the "
            "answer is not contained within the context below, say \"I don't know\". You are not "
            "allowed to answer questions not relevant to the context.\n\n"
            "Context: \n{context}\n\n",
        'it':
            "{identity}\n\n"
            "Rispondi alla domanda nel modo più veritiero possibile utilizzando il contesto fornito e, "
            "se la risposta non è contenuta nel contesto sottostante, dì \"Non lo so\"."
            "Usa esclusivamente informazioni pertinenti al contesto.\n\n"
            "Contesto: \n{context}\n\n"
    }

    @staticmethod
    def knowledge_bases() -> Sequence[str]:
        return next(os.walk('data'))[1]

    @staticmethod
    def set_openai_api_key(openai_api_key: str) -> None:
        openai.api_key = openai_api_key

    def __init__(self,
        knowledge_base: KnowledgeBase,
        tokenizer_encoding: str = 'cl100k_base',
        embeddings_engine: str = 'text-embedding-ada-002',
        model_name: str = 'gpt-3.5-turbo',
        max_prompt_len: int = 1800,
        max_kb_article_len: int = 500,
        max_response_len: int = 200,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.kb = knowledge_base
        if self.kb.language not in self.PROMPT_TEMPLATES:
            raise ValueError(
                f'Unsupported knowledge base langue "{self.kb_language}". '
                f'Supported languages: {", ".join(self.PROMPT_TEMPLATES.keys())}'
            )
        self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        self.embeddings_engine_name = embeddings_engine
        self.model_name = model_name
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.history: List[Mapping[str, str]] = []
        if not self.kb.embeddings_path.is_file():
            self.logger.info(
                'Embeddings file not found, creating embeddings with OpenAI services. '
                'This might take several minutes based on the knowledge base size ...'
            )
            preprocessor = Preprocessor(
                tokenizer_encoding=tokenizer_encoding,
                embeddings_engine=self.embeddings_engine_name,
                max_kb_article_len=max_kb_article_len,
            )
            kb_df = pd.read_csv(self.kb.kb_path)
            kb_lines = (' '.join(row) for row in kb_df.itertuples(index=False))
            with open(self.kb.embeddings_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(Preprocessor.Article._fields)
                writer.writerows(preprocessor.create_embeddings(kb_lines))
            self.logger.info('Embeddings were successfully created.')

        df = pd.read_csv(self.kb.embeddings_path)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        self.embeddings = df
        self.logger.info('Embeddings were successfully loaded.')

    def create_context(self, question: str) -> str:
        '''
        Create a context for a question by finding the most similar context from the embeddings
        '''

        # SEPARATOR = '\n\n###\n\n'
        SEPARATOR = '\n* '

        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(
            input=question,
            engine=self.embeddings_engine_name
        )['data'][0]['embedding']

        # Get the distances from the embeddings
        self.embeddings['distances'] = distances_from_embeddings(
            q_embeddings,
            self.embeddings['embeddings'].values,
            distance_metric='cosine'
        )

        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for _, row in self.embeddings.sort_values('distances', ascending=True).iterrows():

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > self.max_prompt_len:
                break

            # Else add it to the text that is being returned
            returns.append(row['text'])

        # Return the context
        return SEPARATOR.join(returns)

    def answer_question(self,
        question: str,
        stop_sequence: Optional[str] = None,
        temperature: float = 0.0,
        debug=False
    ) -> str:
        """Answer a question based on the most similar context from the kb."""

        context = self.create_context(question)
        self.logger.debug("Context:\n%s", context)
        prompt_template = self.PROMPT_TEMPLATES[self.kb.language]
        prompt = prompt_template.format(
            context=context,
            question=question,
            identity=self.kb.identity
        )

        self.history.append({"role": "user", "content": question})

        messages = [
            {"role": "system", "content": prompt}
        ]
        messages += self.history
        messages += [{
            "role": "system",
            "content": "Ricordati di rispondere alle domande a base del contesto fornito "
                "oppure dì \"Non lo so\"!"
        }]

        if debug:
            print('-' * 20)
            for msg in messages:
                print(msg['role'].upper() + ':')
                print(msg['content'] + '\n')

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
            if debug:
                print('TOKEN_USAGE: prompt={prompt_tokens}, completion={completion_tokens}, '
                      'total={total_tokens}\n'.format(**api_resp['usage']))
            response = api_resp["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            self.logger.exception(e)
            return ""


if __name__ == '__main__':
    import argparse, getpass
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Virtual assistant chatbot")
    parser.add_argument('-kb', '--knowledge_base', type=str, required=True,
                        help="knowledge base (./data subfolder)")
    args = parser.parse_args()

    openai_api_key = getpass.getpass('Enter OpenAI API key: ')

    kb = KnowledgeBase(args.knowledge_base)
    ChatBot.set_openai_api_key(openai_api_key)

    agent = ChatBot(
        knowledge_base=kb,
        max_prompt_len=400,
        max_response_len=400
    )

    print(
        f'\nWelcome to {args.knowledge_base} virtual assistant. Type "exit" to exit. '
        f'Start by typing a question in "{agent.kb.language}" language.'
    )
    while True:
        question = input('\nUser: ')
        if question.lower() == 'exit':
            break
        answer = agent.answer_question(question, debug=True)
        print('\nAssistant:', answer)
