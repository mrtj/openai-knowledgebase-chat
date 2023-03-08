from typing import Optional
import random
import logging

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('meteo')

api_call_word = "METEO_API"
api_resp_word = "METEO_RESPONSE"

prompt = (
        "Sei un'assistente virtuale di previsione meteo che aiuta ai utenti di capire che tempo fa. "
        f"Hai a tua disposizione un API, chiamato {api_call_word}, per sapere le condizioni "
        "meteorologiche in diversi città in tempo reale. Se hai bisogno "
        f"di sapere che tempo fa in una città, scrivi \"{api_call_word}: [nome della città]\". Nel "
        f"prossimo messaggio, METEO_API ti risponderà invece dell'utente con il prefisso \"{api_resp_word}\". "
        "\nPer esempio:\n\n"
        "Utente: Che tempo fa a Debrecen?\n"
        f"Assistente: {api_call_word}: Debrecen\n"
        f"Utente: {api_resp_word}: Condizione meteo a Debrecen: nuvoloso. Temperatura: 13 gradi Celsius.\n"
        "Assistente: Oggi a Debrecen è un fa fresco, ci sono solo 13 gradi. Il cielo è coperto.\n"
    )

messages = []

def add_message(msg):
    global messages
    logger.info(msg)
    messages.append(msg)

add_message({"role": "system", "content": prompt})

def call_api(city: str) -> str:
    logger.info('Calling METEO api with parameters "city=%s" ...', city)
    conditions = [
        'poco nuvoloso', 'molto nuvoloso', 'nuvoloso', 'coperto', 'nubi basse',
        'nubi basse e schiarite', 'parzialmente nuvoloso', 'sereno', 'velature lievi',
        'velature estese', 'soleggiato', 'temporale', 'pioggia'
    ]
    temp: int = random.randint(-5, 35)
    cond: str = conditions[random.randint(0, len(conditions) - 1)]
    api_resp = f'Condizione meteo a {city}: {cond}. Temperatura: {temp} gradi Celsius.'
    logger.info('API response: %s', api_resp)
    return api_resp

def answer_question(
    question: str,
    stop_sequence: Optional[str] = None,
    temperature: float = 0.0
) -> str:

    if question is not None and len(question) > 0:
        add_message({"role": "user", "content": question})

    def complete():
        api_resp = openai.ChatCompletion.create(
            messages=messages,
            temperature=temperature,
            stop=stop_sequence,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model='gpt-3.5-turbo',
        )
        logger.info('TOKEN_USAGE: prompt={prompt_tokens}, completion={completion_tokens}, '
                'total={total_tokens}\n'.format(**api_resp['usage']))
        return api_resp["choices"][0]["message"]["content"].strip()

    try:
        # Create a completions using the question and context
        response = complete()
        add_message({"role": "assistant", "content": response})

        if response.startswith(api_call_word):
            city = response[len(api_call_word):]
            api_resp = call_api(city)
            add_message({"role": "system", "content": f"{api_resp_word}: {api_resp}"})
            response = answer_question(None, stop_sequence=stop_sequence, temperature=temperature)

        return response
    except Exception as e:
        print(e)
        return ""

if __name__ == '__main__':
    import getpass
    openai.api_key = getpass.getpass('Enter OpenAI API key: ')

    print(
        f'\nWelcome to Meteo virtual assistant. Type "exit" to exit. '
        f'Start by typing a question in Italian language.'
    )
    while True:
        question = input('\nUser: ')
        if question.lower() == 'exit':
            break
        answer = answer_question(question)
        print('\nAssistant:', answer)
