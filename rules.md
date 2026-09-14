run or installment:
    pyenv local 3.11
    py -3.11 -m venv .venv
    .venv/scripts/activate
    py -m pip install --upgrade pip
    pip install -r requirements.txt
llm infos read from .env file include:
    url
    key
    model-name