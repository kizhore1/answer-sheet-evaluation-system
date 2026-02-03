# PowerShell helper to create venv and install deps
Set-Location -Path "$PSScriptRoot"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import nltk; nltk.download('punkt')"
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
