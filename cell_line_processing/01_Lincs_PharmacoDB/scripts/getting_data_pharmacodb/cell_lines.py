import json
import requests

url = 'https://pharmacodb.ca/graphql'
allCellLines = {
    'operationName': 'getAllCellLines',
    'query': '''
        query getAllCellLines {
            cell_lines(all: true) {
                id
                uid
                name
                tissue {      id      name      __typename    }
                datasets {      id      name      __typename    }
                __typename
        }
    }
    ''',
    'variables': {}
}

x = requests.post(url, json = allCellLines)
with open(f"cell_lines.json", "w") as f:
    f.write(x.text)
