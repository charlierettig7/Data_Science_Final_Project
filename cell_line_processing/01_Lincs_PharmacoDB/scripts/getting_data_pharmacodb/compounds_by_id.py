import asyncio
import aiohttp
import aiofiles
import copy
import json
import requests

url = 'https://pharmacodb.ca/graphql'
allCompounds = {
    'operationName': 'getAllCompounds',
    'query': '''
        query getAllCompounds {
            compounds(all: true) {
                id
                __typename
            }
        }
    ''',
    'variables': {}
}

x = requests.post(url, json = allCompounds)
ids = []
for compound in json.loads(x.text)["data"]["compounds"]:
    ids.append(compound["id"])

async def do_work(i, session, semaphore):
    async with semaphore:
        singleCompound = {
            'operationName': None,
            'query': '''
                query getSingleCompound($compoundId: Int) {
                    singleCompound: compound(
                        compoundId: $compoundId
                    ) {
                        compound {
                            id
                            name
                            uid
                            annotation {
                                inchikey
                                smiles
                                pubchem
                                fda_status
                                chembl
                                reactome
                                __typename
                            }
                            __typename
                        }
                        __typename
                    }
                }
            ''',
            'variables': {
                'compoundId': i
            }
        }
        singleCompound['variables']['compoundId'] = i
        async with session.post(url, json = singleCompound) as response:
            async with aiofiles.open(f"compound_{i}.json", "w") as f:
                await f.write(await response.text())
        print(f"{i} done")

async def main():
    semaphore = asyncio.Semaphore(30)
    async with aiohttp.ClientSession() as session:
        tasks = [do_work(i, session, semaphore) for i in ids]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
