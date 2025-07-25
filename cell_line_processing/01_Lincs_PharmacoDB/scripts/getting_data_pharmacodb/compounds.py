import json
import requests

url = 'https://pharmacodb.ca/graphql'
allCompounds = {
    'operationName': 'getAllCompounds',
    'query': '''
        query getAllCompounds {
            compounds(all: true) {
                id
                name
                uid
                annotation {
                    pubchem
                    chembl
                    fda_status
                    __typename
                }
                __typename
            }
        }
    ''',
    'variables': {}
}

x = requests.post(url, json = allCompounds)
with open(f"compounds.json", "w") as f:
    f.write(x.text)
