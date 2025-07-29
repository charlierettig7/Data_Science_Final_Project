import json
import requests

url = 'https://pharmacodb.ca/graphql'
myobj = {
    'operationName': None,
    'query': '''
    query getSingleCellLineExperiments($cellLineId: Int!) {
        experiments(cellLineId: $cellLineId) {
            id
            cell_line {      id      uid      name      __typename    }
            compound {      id      uid      name      __typename    }
            tissue {      id      name      __typename    }
            dataset {      id      name      __typename    }
            dose_response {dose      response      __typename    }
            profile {      HS      Einf      EC50      AAC      IC50      DSS1      DSS2      DSS3      __typename    }
            __typename
        }
    }
    ''',
    'variables': {
        'cellLineId': 24
    }
}

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
ids = []
for cl in json.loads(x.text)["data"]["cell_lines"]:
    ids.append(cl["id"])


for i in ids:
    print(i)
    myobj['variables']['cellLineId'] = i
    x = requests.post(url, json = myobj)
    # You can find the output by extracting response_cell_line_i.json.zip
    with open(f"response_cell_line_{i}.json", "w") as f:
        f.write(x.text)
