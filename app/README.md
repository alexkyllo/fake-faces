# FakeFace App
## Description
There are two projects in this folder.
### Fakefaceapp
- This is the front end application. Written in Vue, it calls an API with the URL of the included image as a query string.

### Classify
- This is an Azure function which serves as the API for the model.

## Running the App
### Prerequisites
1. [Azure Functions core tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#install-the-azure-functions-core-tools)
2. [Python 3.8.6](https://www.python.org/downloads/release/python-386/)

### Instructions
1. From the terminal - start the function by activating the virtual environment: ```.venv/Scripts/activate```
2. Start the local function: ```func start```
3. Change to the fakefaceapp folder and start the local dev server: ```npm run serve```
4. Drop the url to an image into the App.

Test with: https://upload.wikimedia.org/wikipedia/commons/9/98/Tom_Hanks_face.jpg



