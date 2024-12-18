"""
# At Command Prompt:
    D:\DataBackup\Work\2021-11-15 Kalyana@Wells Fargo\2024-03-02 ResumeMatch\04_Deployment>start /b uvicorn Simple:App --reload
    D:\DataBackup\Work\2021-11-15 Kalyana@Wells Fargo\2024-03-02 ResumeMatch\04_Deployment>uvicorn SimpleHttps:App --ssl-keyfile "D:\DataBackup\Work\2021-11-15 Kalyana@Wells Fargo\2024-03-02 ResumeMatch\04_Deployment\key.pem" --ssl-certfile "D:\DataBackup\Work\2021-11-15 Kalyana@Wells Fargo\2024-03-02 ResumeMatch\04_Deployment\cert.pem"

# In Browser:
    http://127.0.0.1:8000
    
# Note(s):
    * The data files to be uploaded for should be in the same directory as the 'SimpleHttps.py' file.
"""


#--------------------------------------------------------------------------------#
# Import necessary modules.
#--------------------------------------------------------------------------------#
# FastAPI related imports.
from fastapi import FastAPI
from fastapi import Request
from fastapi import File
from fastapi import UploadFile

# Jinja2Templates related imports.
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import ssl

# Basic imports.
import os
import re


#--------------------------------------------------------------------------------#
# Define Constants.
#--------------------------------------------------------------------------------#
# Path(s) and File(s).
TEXT_FILE_EXT_TXT = r".txt"
TEMPLATE_PATH = r"Templates"
INPUT_HTML_FILE = r"Simple_Input.html"
OUTPUT_HTML_FILE = r"Simple_Output.html"


#--------------------------------------------------------------------------------#
# Define Global Variables.
#--------------------------------------------------------------------------------#
# Object of FastAPI.
App = FastAPI()

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("cert.pem", keyfile="key.pem")

# Templates.
Templates = Jinja2Templates(directory = TEMPLATE_PATH)


#--------------------------------------------------------------------------------#
# End Points.
#--------------------------------------------------------------------------------#
# GET option.
@App.get('/')
async def root(request:Request):
    # return {"message": "Hello World!"}
    return Templates.TemplateResponse(INPUT_HTML_FILE, context = {'request':request})


# POST option.
@App.post('/')
async def root(request:Request, UserFile: UploadFile = File(...)):
    MyList = list()
    Total = 0
    
    # Get the filename of the uploaded ASCII text data file.
    UserFileName = UserFile.filename
    # return {"UserFileName": UserFileName}
    
    # Extract the file extension from the file name.
    UserFileExtn = UserFileName[-4:]

    # Check the file extension to take appropriate action.
    if UserFileExtn == TEXT_FILE_EXT_TXT:
        # If the user uploaded file is of specific type then only process the file.
        with open(UserFileName, 'r') as Fptr:
            # Iterate through each line of the input data file.
            for Line in Fptr:
                MyList.append(Line)
                
        for Idx in range(len(MyList)):
            Total = Total + int(MyList[Idx])
        
        # Return the HTML page that can display the input and output videos.
        return Templates.TemplateResponse(OUTPUT_HTML_FILE, context = {'request':request, "ClassProb":Total})
    else:
        return {"Return":"Unsupported File Selected."}


#--------------------------------------------------------------------------------#
# Main Code.
#--------------------------------------------------------------------------------#
# if __name__ == "__main__":
