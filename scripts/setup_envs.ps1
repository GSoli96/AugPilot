# Ricrea gli ambienti di AugPilot (envs/<nome>) dai file di lock in envs_lock/ (WP5 di the project plan).
# Usa uv: installa Python 3.10 se manca e installa le versioni esatte del lock con --no-deps
# (il lock è un pip freeze completo, quindi le dipendenze sono già tutte elencate).
#
# Uso, dalla cartella del progetto:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1                 # tutti
#   powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1 -Envs graph,ts  # solo alcuni
# Idempotente: un ambiente che importa già i suoi pacchetti chiave non viene reinstallato.
param(
    [string[]]$Envs = @('tabular', 'argn', 'ts', 'ts_tf', 'graph'),
    [string]$Python = '3.10'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root 'envs_setup.log'
function Log($m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; Write-Output $line; Add-Content -Path $log -Value $line -Encoding utf8 }

# 1. uv
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $cand = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path $cand)) {
        Log 'uv non trovato: installo con lo script ufficiale di astral.sh'
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
    }
    $uv = $cand
}
if (-not (Test-Path $uv)) { Log "ERRORE: uv non disponibile"; exit 2 }
Log "uv: $(& $uv --version)"

# 2. Python 3.10 gestito da uv
& $uv python install $Python 2>&1 | ForEach-Object { Log $_ }

# Pacchetti chiave per il controllo finale (import)
$check = @{
    tabular = 'import torch, sklearn, imblearn, ctgan, xgboost; print(torch.__version__, torch.cuda.is_available())'
    argn    = 'import torch, mostlyai.engine; print(torch.__version__, torch.cuda.is_available())'
    ts      = 'import torch, tsgm; print(torch.__version__, torch.cuda.is_available())'
    ts_tf   = 'import tensorflow as tf; print(tf.__version__)'
    graph   = 'import torch, torch_geometric, torch_scatter; print(torch.__version__, torch.cuda.is_available())'
}

$failed = @()
# Con -File una lista "a,b,c" arriva come stringa unica: la divido sulle virgole.
$Envs = @($Envs | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($e in $Envs) {
    $lock = Join-Path $root "envs_lock\$e.txt"
    $py = Join-Path $root "envs\$e\Scripts\python.exe"
    if (-not (Test-Path $lock)) { Log "ERRORE: manca $lock"; $failed += $e; continue }
    if (Test-Path $py) {
        $ok = & $py -c $check[$e] 2>$null
        if ($LASTEXITCODE -eq 0) { Log "$e già pronto ($ok): salto"; continue }
    }
    Log "${e}: creo il venv e installo $((Get-Content $lock).Count) pacchetti"
    & $uv venv (Join-Path $root "envs\$e") --python $Python --seed 2>&1 | ForEach-Object { Log $_ }
    $uvArgs = @('pip', 'install', '--python', $py, '--no-deps', '-r', $lock,
              '--index-strategy', 'unsafe-best-match',
              '--extra-index-url', 'https://download.pytorch.org/whl/cu124',
              '--find-links', 'https://data.pyg.org/whl/torch-2.6.0+cu124.html')
    & $uv @uvArgs 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: installazione di $e fallita"; $failed += $e; continue }
    $ok = & $py -c $check[$e] 2>&1
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: controllo di $e fallito: $ok"; $failed += $e } else { Log "$e OK: $ok" }
}
if ($failed.Count -gt 0) { Log "FINE con errori: $($failed -join ', ')"; exit 1 }
Log 'FINE: tutti gli ambienti pronti'
exit 0
