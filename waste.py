from google.cloud import vision
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file('./secrets/jeemathtutoragent-51b1906d4eef.json')
client = vision.ImageAnnotatorClient(credentials=creds)
print('Vision client OK')
