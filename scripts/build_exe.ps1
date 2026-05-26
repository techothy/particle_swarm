# Build Windows executable for PSO-TF-IDF GUI
# Run from repo root: .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

& .\.venv\Scripts\pip.exe install -r requirements-gui.txt -q
& .\.venv\Scripts\pip.exe install -e . -q

$Out = Join-Path $Root "dist"
Write-Host "Building executable (this may take several minutes)..."
& .\.venv\Scripts\pyinstaller.exe `
    --noconfirm `
    --windowed `
    --name "PSO-TF-IDF" `
    --paths (Join-Path $Root "src") `
    --collect-all sklearn `
    --collect-all nltk `
    --hidden-import "statsmodels" `
    --hidden-import "PIL" `
    --hidden-import "customtkinter" `
    (Join-Path $Root "gui\app.py")

Write-Host ""
Write-Host "Done. Run: $Out\PSO-TF-IDF\PSO-TF-IDF.exe"
Write-Host "First launch may download 20 Newsgroups data; allow network access."
