from django.http import JsonResponse

from dotenv import load_dotenv

from google.oauth2 import id_token
from google.auth.transport import requests
import os
from db_tools import get_user_by_email
load_dotenv()

is_local = os.environ.get("IS_LOCAL")=="True"
def verify_token(request,get_token=False):
    try:
        if is_local!= True:
            token=request.headers['Authorization'].split(" ")[1]
            decoded_token = id_token.verify_oauth2_token(token, requests.Request())
        if get_token:
            return decoded_token
    except:
        return "failed"

def get_user_id(request):
    if is_local:
        user_id = 1
        return user_id
    else:
            #get the user from the token
            decoded_token = verify_token(request,get_token=True)
            if decoded_token=="failed":
                return "failed"
            user_id=get_user_by_email(decoded_token['email'])
            return user_id