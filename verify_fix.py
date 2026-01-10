import requests
from io import BytesIO

csv_content = b"ip,click_hour,user_agent\n192.168.1.1,10,Mozilla"
files = {'file': ('test.csv', BytesIO(csv_content), 'text/csv')}

try:
    response = requests.post('http://127.0.0.1:8000/predict', files=files, timeout=5)
    print(f"Status Code: {response.status_code}")
    # print(f"Response: {response.text}")
    if response.status_code == 200:
        print("SUCCESS: Endpoint returned 200 OK")
    else:
        print(f"FAILURE: Endpoint returned {response.status_code}")
except Exception as e:
    print(f"Error connecting to server: {e}")
