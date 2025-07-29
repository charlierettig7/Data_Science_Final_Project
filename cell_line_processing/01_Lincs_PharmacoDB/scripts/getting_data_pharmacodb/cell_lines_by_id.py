import asyncio
import aiohttp
import aiofiles
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
ids = []
for cl in json.loads(x.text)["data"]["cell_lines"]:
    ids.append(cl["id"])

async def do_work(i, session, semaphore):
    async with semaphore:
        myobj = {
            'operationName': None,
            'query': '''
                query getSingleCellLine($cellId: Int) {
                    cell_line(cellId: $cellId) {
                        id
                        uid
                        name
                        diseases
                        accession_id
                        tissue {
                            id
                            name
                            __typename
                        }
                        synonyms {
                            name
                            dataset {
                                id
                                name
                                __typename
                            }
                            __typename
                        }
                        __typename
                    }
                }
            ''',
            'variables': {
                'cellId': i
            }
        }
        async with session.post(url, json = myobj) as response:
            # You can find the output by extracting response_cell_line_i.json.zip
            async with aiofiles.open(f"cell_line_by_id_{i}.json", "w") as f:
                await f.write(await response.text())
        print(f"{i} done")

async def main():
    semaphore = asyncio.Semaphore(30)
    async with aiohttp.ClientSession() as session:
        tasks = [do_work(i, session, semaphore) for i in ids]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
