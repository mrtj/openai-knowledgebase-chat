from typing import Optional, Sequence, Mapping, Tuple
import logging
import enum
import re
import json

import openai
import requests

class MessageType(enum.Enum):
    SYSTEM: str = 'system'
    USER: str = 'user'
    ASSISTANT: str = 'assistant'

class MeteoChatbot:

    API_CALL_WORD = "METEO_API"
    API_RESP_WORD = "METEO_RESPONSE"

    PROMPT = (
            "Sei un'assistente virtuale di previsione meteo che aiuta ai utenti di capire che tempo fa. "
            f"Hai a tua disposizione un API, chiamato {API_CALL_WORD}, per sapere le condizioni "
            "meteorologiche in diversi città in tempo reale. Se hai bisogno "
            f"di sapere che tempo fa in una città, scrivi \"{API_CALL_WORD}: [nome della città]\". Nel "
            f"prossimo messaggio, METEO_API ti risponderà invece dell'utente con il prefisso \"{API_RESP_WORD}\". "
            "\nPer esempio:\n\n"
            "Utente: Che tempo fa a Debrecen?\n"
            f"Assistente: {API_CALL_WORD}: Debrecen\n"
            f"Utente: {API_RESP_WORD}: Condizione meteo a Debrecen: nuvoloso. Temperatura: 13 gradi Celsius.\n"
            "Assistente: Oggi a Debrecen è un fa fresco, ci sono solo 13 gradi. Il cielo è coperto.\n"
        )

    WEATHER_CODES = {
        0: 'sereno',
        1: 'poco nuvoloso',
        2: 'parzialmente nuvoloso',
        3: 'coperto',
        45: 'nebbia',
        48: 'brina',
        51: 'pioviggina leggera',
        53: 'pioviggina moderata',
        55: 'pioviggina densa',
        56: 'pioviggina gelata leggera',
        57: 'pioviggina gelata densa',
        61: 'pioggia leggera',
        63: 'pioggia moderata',
        63: 'pioggia intensa',
        66: 'pioggia gelida leggera',
        67: 'pioggia gelida densa',
        71: 'nevicata debole',
        73: 'nevicata moderata',
        75: 'nevicata intensa',
        77: 'granelli di neve',
        80: 'rovesci di pioggia leggeri',
        81: 'rovesci di pioggia moderati',
        82: 'rovesci di pioggia violenti',
        85: 'rovesci di pioggia deboli',
        85: 'rovesci di pioggia intensi',
        95: 'temporale',
        96: 'temporale con lieve grandinata',
        99: 'temporale con forte grandinata',
    }

    messages: Sequence[Mapping[str, str]] = []
    call_word_regex = r'METEO_API:\s+([a-zA-Z ]*).*$'

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.add_message(MessageType.SYSTEM, self.PROMPT)

    def add_message(self, type_: MessageType, msg: str) -> None:
        self.messages.append({'role': type_.value, 'content': msg})

    def geocode_api(self, city) -> Tuple[Optional[float], Optional[float]]:
        geocode_url = 'https://geocoding-api.open-meteo.com/v1/search'
        params = {'name': city, 'count': 1}
        resp = requests.get(geocode_url, params=params)
        resp.raise_for_status()
        resp = resp.json()
        self.logger.debug('Geocode API response: %s', json.dumps(resp, indent=4))
        resp = resp.get('results', [])
        resp = resp[0] if len(resp) > 0 else {}
        return (resp.get('latitude'), resp.get('longitude'))

    def weather_api(self, city: str, lat: float, lon: float):
        weather_url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': lat,
            'longitude': lon,
            'current_weather': 'true',
        }
        resp = requests.get(weather_url, params=params)
        resp.raise_for_status()
        resp = resp.json()
        self.logger.debug('Meteo API response: %s', json.dumps(resp, indent=4))
        resp = resp['current_weather']
        code = int(resp['weathercode'])
        temp = float(resp['temperature'])
        cond = self.WEATHER_CODES.get(code, 'sconosciuta')
        api_resp = f'Condizione meteo a {city}: {cond}. Temperatura: {temp} gradi Celsius.'
        return api_resp

    def call_api(self, city: str) -> str:
        ''' Simulates an external API call. '''
        self.logger.info('Calling geocoding api with parameters "city=%s" ...', city)
        lat, lon = self.geocode_api(city)
        if lat is None or lon is None:
            api_resp = f'ERROR: Unknown city "{city}"'
            self.logger.warning('Meteo API error: %s', api_resp)
            return api_resp
        self.logger.info('Geocode results: lat=%f, lon=%f', lat, lon)
        self.logger.info('Calling meteo api with parameters "lat=%f, lon=%f"', lat, lon)
        api_resp = self.weather_api(city, lat, lon)
        self.logger.info('Meteo API response: %s', api_resp)
        return api_resp

    def answer_question(self,
        question: str,
        stop_sequence: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:

        if question is not None and len(question) > 0:
            self.add_message(MessageType.USER, question)

        def complete():
            api_resp = openai.ChatCompletion.create(
                messages=self.messages,
                temperature=temperature,
                stop=stop_sequence,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model='gpt-3.5-turbo',
            )
            self.logger.info('TOKEN_USAGE: prompt={prompt_tokens}, completion={completion_tokens}, '
                    'total={total_tokens}\n'.format(**api_resp['usage']))
            return api_resp["choices"][0]["message"]["content"].strip()

        try:
            # Create a completions using the question and context
            response = complete()
            self.add_message(MessageType.ASSISTANT, response)

            matches = re.findall(self.call_word_regex, response, re.MULTILINE)
            if matches:
                full_response = ''
                first_response = re.sub(self.call_word_regex, '', response, 0, re.MULTILINE).strip()
                if first_response:
                    full_response = first_response + ' '
                for city in matches:
                    api_resp = self.call_api(city)
                    self.add_message(MessageType.SYSTEM, f"{MeteoChatbot.API_RESP_WORD}: {api_resp}")

                response = (
                    full_response +
                    self.answer_question(None, stop_sequence=stop_sequence, temperature=temperature)
                )

            return response
        except Exception as e:
            print(e)
            return ""

if __name__ == '__main__':
    import getpass, pprint, os
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.environ.get('OPENAI_API_KEY') or getpass.getpass('Enter OpenAI API key: ')
    chatbot = MeteoChatbot()
    logging.basicConfig(level=logging.INFO)

    print(
        '\nWelcome to Meteo virtual assistant. Type "exit" to exit, "debug" to show current '
        'chat history. Start by typing a question in Italian language.'
    )
    while True:
        question = input('\nUser: ')
        if question.lower() == 'exit':
            break
        elif question.lower() == 'debug':
            pprint.pprint(chatbot.messages)
            continue
        else:
            answer = chatbot.answer_question(question)
            print('\nAssistant:', answer)
