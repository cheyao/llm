# Ascendance of a bookworm LLM
Making a llm for the hackclub ysws program

I'm using 33 volumes of Ascendance of a Bookworm as training data :D

## Usage

```
python3 -m env vnev
source vnev/bin/activate
pip install -r requirements.txt
curl -L https://www.kaggle.com/api/v1/models/geminn/ascendance-of-a-bookworm-llm/pyTorch/default/1/download
fastapi run main.py
```
Now visit http://localhost:8000/

## Disclaimer
This project is just a educational project, if any of the publishers isn't happy, just email me a message and I'll sort it out.
