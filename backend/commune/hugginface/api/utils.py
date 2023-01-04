import aiohttp

GRADIO_SPACES_API_URL = "https://$SPACEHOST.hf.space"


def searchConfigForApiName(data):
        if not "dependencies" in data:
            return None
        else:
            for dependencies in data["dependencies"]:
                if "api_name" in dependencies:
                    return 200
        return None

def gradioSDK(data=[], endpoint="/"):
    endpoint.startswith("/")
    return [f'{GRADIO_SPACES_API_URL}{endpoint}'.replace("$SPACEHOST", str(space["id"]).replace('/', '-')) for space in data if "sdk" in space and space["sdk"] == "gradio"]


def statusStrategy(resposne : aiohttp.ClientResponse):
    if resposne.status == 200:
        return True
        
            