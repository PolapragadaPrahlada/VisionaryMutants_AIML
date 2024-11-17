#--------------------------------------------------------------------------------#
# Import necessary modules.
#--------------------------------------------------------------------------------#
# Basic imports.
import os
import re

# NLTK library imports.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# FastAPI related imports.
from fastapi import FastAPI
from fastapi import Request
from fastapi import File
from fastapi import UploadFile

# Jinja2Templates related imports.
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Text Representation related imports.
from sklearn.feature_extraction.text import TfidfVectorizer

# Save and Load modules as pickel files.
import pickle


#--------------------------------------------------------------------------------#
# Define Constants.
#--------------------------------------------------------------------------------#
# Path(s) and File(s).
ROOT_PATH = r"C:\Users\kalya\Desktop\RVHackathon\2024-03-02 ResumeMatch"
MODEL_FILE_PATH = r"C:\Users\kalya\Desktop\RVHackathon\2024-03-02 ResumeMatch\03_Output"
MODEL_FILE_NAME = r"Stacking.pickle"
VECTOR_FILE_NAME = r"TfIdf.pickle"
TEMPLATE_PATH = r"Templates"
INPUT_HTML_FILE = r"FrontEnd_Input.html"
OUTPUT_HTML_FILE = r"FrontEnd_Output.html"
TEXT_FILE_EXT_TXT = r".txt"

# Keep the words having length greater than or equal to this value.
WORD_LENGTH_TO_KEEP = 3


#--------------------------------------------------------------------------------#
# Define Global Variables.
#--------------------------------------------------------------------------------#
# Object of FastAPI.
App = FastAPI()

# Templates.
Templates = Jinja2Templates(directory = TEMPLATE_PATH)

# Special characters.
SpecialcharList = [
                    '!', '"', '#', '$', '%', 
                    '&', "'", '(', ')', '*', 
                    ',', '-', '.', '/', ':', 
                    ';', '<', '=', '>', '?', 
                    '@', '[', '\\', ']', '^', 
                    '_', '`', '{', '|', '}', 
                    '~',
                  ]

# Class labels in the order of 'ClassLabelList'.
ClassLabelList = [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 ]

# Class names in the order of 'ClassLabelList'.
ClassNameList = [
                    'Automation Developer',
                    'ETL Developer',
                    'Java Developer',
                    'Python Developer',
                    'SAP Developer',
                    'Blockchain Engineer',
                    'DataScience Engineer',
                    'DevOps Engineer',
                    'Hadoop Engineer',
                    'WebDesign Engineer',
                    'Advocate',
                    'Business Analyst',
                    'Civil Engineer',
                    'HR Manager',
                    'Sales Person',
                ]

# Vector and model files.
TfIdfObject = 0
ModelStack = 0


#--------------------------------------------------------------------------------#
# User Defined Functions (UDFs).
#--------------------------------------------------------------------------------#
def GetLowerCaseText(Text):
    Text = Text.lower()
    return Text


def GetNonasciiFreeText(Text):
    Pattern = re.compile(r'[^\x00-\x7F]+', flags=re.I)
    Text = re.sub(Pattern, ' ', Text)
    return Text


def GetSpecialcharFreeText(Text):
    for Specialchar in SpecialcharList:
        Text = Text.replace(Specialchar, ' ')
    return Text


def RemoveNonalphabetWords(Text):
    Text = ' '.join([Word for Word in Text.split() if Word.isalpha()])
    return Text


def RemoveStopWords(Text):
    StopWords = set(stopwords.words('english'))
    Text = ' '.join([Word for Word in Text.split() if Word not in StopWords])
    return Text


def KeepWordsofLength(Text):
    Text = ' '.join([Word for Word in Text.split() if len(Word) >= WORD_LENGTH_TO_KEEP])
    return Text


def GetWhitespaceFreeText(Text):
    Pattern = re.compile(r'\s+', flags=re.I)
    Text = re.sub(Pattern, ' ', Text)
    Text = re.sub(r'\s+', ' ', Text, flags=re.I)
    return Text


def CleanupText(Text, *FunctionNames):
    if len(FunctionNames):
        for FunctionName in FunctionNames:
            Text = FunctionName(Text)
    return Text


def ManageInput(FullPathAndFile):
    Text = ""
    with open(FullPathAndFile, 'r') as Fptr:
        for Line in Fptr:
            Text = Text + ' ' + Line
    return Text


def PredictText(Model, Text):
    Text = CleanupText(Text, GetLowerCaseText, GetNonasciiFreeText, \
                       GetSpecialcharFreeText, RemoveNonalphabetWords, \
                       RemoveStopWords, KeepWordsofLength, \
                       GetWhitespaceFreeText)
    TfidfTestVectors = TfIdfObject.transform([Text])
    Prediction = Model.predict(TfidfTestVectors)[0]
    Cp = Model.predict_proba(TfidfTestVectors)[0]
    return Prediction, Cp


def ProcessUserFile(UserFile):
    """
    User Defined Function (UDF) to handle the uploaded ASCII text data file.
    Input: Uploaded ASCII text data file.
    Output: Prediction String, Class Probability (CP) with assumption.
    """
    global TfIdfObject
    global ModelStack

    Text = ManageInput(UserFile.filename)

    TmpPath = MODEL_FILE_PATH + r"\\" + VECTOR_FILE_NAME
    with open(TmpPath, 'rb') as PickleFile:
        TfIdfObject = pickle.load(PickleFile)

    TmpPath = MODEL_FILE_PATH + r"\\" + MODEL_FILE_NAME
    with open(TmpPath, 'rb') as PickleFile:
        ModelStack = pickle.load(PickleFile)

    LabelVsNameDict = dict(zip(ClassLabelList, ClassNameList))
    Prediction, Cp = PredictText(ModelStack, Text)

    PredictionStr = "This resume is suitable for the position: {:s}".format(LabelVsNameDict[Prediction])
    confidence_score = Cp[Prediction] * 100

    if confidence_score < 25:
        assumption = "ASSUMING FAKE RESUME"
    else:
        assumption = "ASSUMING TRUE RESUME"

    ClassProb = "Confidence Score: {:02.2f}% ({})".format(confidence_score, assumption)

    return str(PredictionStr), str(ClassProb)


#--------------------------------------------------------------------------------#
# End Points.
#--------------------------------------------------------------------------------#
@App.get('/')
async def root(request:Request):
    return Templates.TemplateResponse(INPUT_HTML_FILE, context = {'request':request})


@App.post('/')
async def root(request:Request, UserFile: UploadFile = File(...)):
    UserFileName = UserFile.filename
    UserFileExtn = UserFileName[-4:]

    if UserFileExtn == TEXT_FILE_EXT_TXT:
        PredictionStr, ClassProb = ProcessUserFile(UserFile)
        return Templates.TemplateResponse(OUTPUT_HTML_FILE, context = {'request':request, "PredictionStr":PredictionStr, "ClassProb":ClassProb})
    else:
        return {"Return":"Unsupported File Selected."}
