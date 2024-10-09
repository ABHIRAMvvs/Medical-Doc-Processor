import requests
url = "http://162.19.116.128:8000/process_pdf/"
pdf_file = {'pdf': open("./test.pdf", 'rb')}
response = requests.post(url, files=pdf_file)
print(response.json())