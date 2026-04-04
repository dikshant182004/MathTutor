## To start our streamlit app manually via powershell from the project root 

$env:PYTHONPATH = "$PSScriptRoot\src"
streamlit run "$PSScriptRoot\src\frontend\app.py"