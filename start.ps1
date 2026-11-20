# Run from PowerShell in a git checkout. The current desktop setup uses a local virtual environment.
$ErrorActionPreference = 'Stop'
$forecastPython = $null
$candidates = @(
    (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    (Join-Path $env:LOCALAPPDATA 'aiu-grade-forecast/venv/Scripts/python.exe')
)
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $forecastPython = $candidate
        break
    }
}
if (-not $forecastPython) {
    throw 'Create .venv and install requirements.txt as described in README.md.'
}
& $forecastPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1 --browser.gatherUsageStats false
exit $LASTEXITCODE
