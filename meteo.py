from typing import Optional, Sequence, Mapping, Tuple
import logging
import enum
import re
import json

import openai
import requests

from chat_tools import Language, MessageType

class MeteoChatbot:

    class Template(enum.Enum):
        prompt = 1,
        meteo_api_response = 2

    API_CALL_WORD = "METEO_API"
    API_RESP_WORD = "METEO_RESPONSE"

    TEMPLATES = {
        Language.EN: {
            Template.prompt:
                f"You are a weather forecasting virtual assistant that helps users understand what "
                f"the weather is like. You can use an API, called {API_CALL_WORD}, to figure out the "
                f"weather conditions in different cities in real time. If you need to know what is the "
                f"weather like in a city, write \"{API_CALL_WORD}: [city name]\". In the next message, "
                f"{API_CALL_WORD} will reply to you instead of the user with the prefix "
                f"\"{API_RESP_WORD}\".\n"
                f"For example:\n\n"
                f"User: Is it cold today in Debrecen?\n"
                f"Assistant: METEO_API: Debrecen\n"
                f"System: METEO_RESPONSE: Weather in Debrecen: Cloudy. Temperature: 13 degrees Celsius.\n"
                f"Assistant: Today it's cool in Debrecen, it's only 13 degrees. The sky is cloudy.\n",
            Template.meteo_api_response:
                "Weather conditions in {city}: {cond}. Temperature: {temp} degree Celsius."
        },
        Language.IT: {
            Template.prompt:
                f"Sei un'assistente virtuale di previsione meteo che aiuta ai utenti di capire che "
                f"tempo fa. Hai a tua disposizione un API, chiamato {API_CALL_WORD}, per sapere le "
                f"condizioni meteorologiche in diversi città in tempo reale. Se hai bisogno di sapere "
                f"che tempo fa in una città, scrivi \"{API_CALL_WORD}: [nome della città]\". Nel "
                f"prossimo messaggio, METEO_API ti risponderà invece dell'utente con il prefisso "
                f"\"{API_RESP_WORD}\".\n"
                f"Per esempio:\n\n"
                f"Utente: Che tempo fa a Debrecen?\n"
                f"Assistente: {API_CALL_WORD}: Debrecen\n"
                f"Sistema: {API_RESP_WORD}: Condizione meteo a Debrecen: nuvoloso. Temperatura: 13 gradi Celsius.\n"
                f"Assistente: Oggi a Debrecen è un fa fresco, ci sono solo 13 gradi. Il cielo è coperto.\n",
            Template.meteo_api_response:
                "Condizione meteo a {city}: {cond}. Temperatura: {temp} gradi Celsius."
        }
    }

    WEATHER_CODES = {

        Language.IT: {
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
            65: 'pioggia intensa',
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
            -1: 'sconosciuta'
        },
        Language.EN: {
            0: 'clear sky',
            1: 'mainly clear',
            2: 'partly cloudy',
            3: 'overcast',
            45: 'fog',
            48: 'depositing rime fog',
            51: 'light drizzle',
            53: 'moderate drizzle',
            55: 'intense drizzle',
            56: 'light freezing drizzle',
            57: 'dense freezing drizzle',
            61: 'slight rain',
            63: 'moderate rain',
            65: 'heavy rain',
            66: 'light freezing rain',
            67: 'heavy freezing rain',
            71: 'slight snow fall',
            73: 'moderate snow fall',
            75: 'heavy snow fall',
            77: 'snow grains',
            80: 'slight rain shower',
            81: 'moderate rain shower',
            82: 'heavy rain shower',
            85: 'slight snow shower',
            85: 'heavy snow shower',
            95: 'thunderstorm',
            96: 'thunderstorm with slight hail',
            99: 'thunderstorm with heavy hail',
            -1: 'unknown'
        }
    }

    messages: Sequence[Mapping[str, str]] = []
    call_word_regex = re.compile(f'{API_CALL_WORD}:\s+([a-zA-Z ]*).*$', re.MULTILINE)

    def __init__(self, language: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        languages = [lang.name for lang in self.TEMPLATES.keys()]
        if language.upper() not in languages:
            raise ValueError(
                f'Unsupported knowledge base langue "' + language + '". '
                'Supported languages: ' + ", ".join(languages)
            )
        self.language = Language[language.upper()]
        self.template = self.TEMPLATES[self.language]
        self.weather_code = self.WEATHER_CODES[self.language]
        self.add_message(MessageType.SYSTEM, self.template[MeteoChatbot.Template.prompt])

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
        cond = self.weather_code.get(code, self.weather_code[-1])
        api_resp = (
            self.template[MeteoChatbot.Template.meteo_api_response]
            .format(city=city, cond=cond, temp=temp)
        )
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

        def complete() -> str:
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

            matches = self.call_word_regex.findall(response)
            if matches:
                full_response = ''
                api_called = False
                for city in matches:
                    if not city:
                        continue
                    api_resp = self.call_api(city)
                    self.add_message(MessageType.SYSTEM, f"{MeteoChatbot.API_RESP_WORD}: {api_resp}")
                    api_called = True
                if not api_called:
                    return response
                first_response = self.call_word_regex.sub('', response).strip()
                if first_response:
                    full_response = first_response + ' '
                response = (
                    full_response +
                    self.answer_question(None, stop_sequence=stop_sequence, temperature=temperature)
                )

            return response
        except Exception as e:
            print(e)
            return ""

if __name__ == '__main__':
    import getpass, pprint, os, argparse
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Meteo chatbot")
    languages = [lang.name for lang in MeteoChatbot.TEMPLATES.keys()]
    parser.add_argument('-l', '--language', type=str, choices=languages,
                        help='chatbot language', default='en')
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.environ.get('OPENAI_API_KEY') or getpass.getpass('Enter OpenAI API key: ')
    chatbot = MeteoChatbot(language=args.language)
    logging.basicConfig(level=logging.INFO)

    print(
        '\nWelcome to Meteo virtual assistant. Type "exit" to exit, "debug" to show current '
        f'chat history. Start by typing a question in "{args.language}" language.'
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
