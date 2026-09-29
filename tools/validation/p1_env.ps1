param([ValidateSet('cpu', 'cuda')][string]$Mode = 'cpu')
$env:RUSTUP_AUTO_INSTALL = '0'
$env:RUSTUP_TOOLCHAIN = '1.98.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:VSLANG = '1033'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:CARGO_HTTP_PROXY = 'http://127.0.0.1:7897'
$ErrorActionPreference = 'Stop'
$script:P1Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script:P1Cargo = (rustup which cargo --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 Cargo is unavailable' }
$env:RUSTC = (rustup which rustc --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 rustc is unavailable' }
$env:RUSTDOC = (rustup which rustdoc --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 rustdoc is unavailable' }
$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-msvc'
if ($Mode -eq 'cpu') {
    Get-ChildItem Env: | Where-Object Name -Match '^(CUDA|CUDNN)' | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
    }
    $env:PATH = (($env:PATH -split ';') | Where-Object { $_ -notmatch 'NVIDIA GPU Computing Toolkit' }) -join ';'
    $env:CARGO_TARGET_DIR = Join-Path $script:P1Root 'target/p1-cpu'
} else {
    $required = @{ CL = '/Zc:preprocessor /utf-8'; NVCC_APPEND_FLAGS = '-Xcompiler=/MT'; RUSTFLAGS = '-C target-feature=+crt-static' }
    foreach ($key in $required.Keys) {
        $existing = [Environment]::GetEnvironmentVariable($key, 'Process')
        if ($existing -and $existing -ne $required[$key]) { throw "Inspect conflicting $key before using the P0 CUDA cache" }
    }
    $vs = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
    $entries = & $env:ComSpec /d /c ('call "{0}" >nul && set' -f $vs)
    if ($LASTEXITCODE -ne 0) { throw 'MSVC initialization failed' }
    foreach ($entry in $entries) {
        if ($entry -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    $env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
    $env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH"
    foreach ($key in $required.Keys) { [Environment]::SetEnvironmentVariable($key, $required[$key], 'Process') }
    $env:CARGO_TARGET_DIR = Join-Path $script:P1Root 'target/windows-static'
    $env:VSLANG = '1033'
}
function Get-P1LogPath {
    $directory = Join-Path $script:P1Root 'runs/p1-development'
    [void][IO.Directory]::CreateDirectory($directory)
    Join-Path $directory ((Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.log')
}
function Invoke-P1Cargo {
    $log = Get-P1LogPath
    ('cargo ' + ($args -join ' ')) | Set-Content -LiteralPath $log -Encoding utf8
    & $script:P1Cargo @args 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    Write-Output "P1 validation log: $log; exit_code=$code"
    if ($code -ne 0) { throw 'Cargo failed; stop the current step' }
}
function Invoke-P1Reference {
    $log = Get-P1LogPath
    ('uv run --frozen --project tools/reference python ' + ($args -join ' ')) | Set-Content -LiteralPath $log -Encoding utf8
    uv run --frozen --project tools/reference python @args 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    Write-Output "P1 validation log: $log; exit_code=$code"
    if ($code -ne 0) { throw 'Reference check failed; stop the current step' }
}
