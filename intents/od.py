import requests
import json

def get_od_synonyms(word):
    word = word
    app_id = '9eff540e'
    app_key = '7cd7fa79b6c76d9b55c11e2caf0eae60'
    language = 'en'
    word_id = 'example'
    url = f"https://od-api.oxforddictionaries.com:443/api/v2/thesaurus/en/{word}?fields=synonyms&strictMatch=false"
    # (f"https://od-api.oxforddictionaries.com/api/v2/thesaurus/{language}/{word_id.lower()}")
    # "https://od-api.oxforddictionaries.com/api/v2/inflections/en/jump"


    

    response = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})
    json_data = json.loads(response.text)

    if 'results' in json_data:
        for result in json_data['results']:
            for lexicalEntry in result['lexicalEntries']:
                for entry in lexicalEntry['entries']:
                    for sense in entry['senses']:
                        if 'synonyms' in sense:
                            synonyms = [syn['text'] for syn in sense['synonyms']]
                            # print(f"Synonyms for {word_id}: {', '.join(synonyms)}")
                            return synonyms
