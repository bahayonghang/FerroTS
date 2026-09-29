param([Parameter(Mandatory=$true)][string]$Suite)
$env:RUSTUP_AUTO_INSTALL = '0'
$env:RUSTUP_TOOLCHAIN = '1.98.0'
$env:PYTHONIOENCODING = 'utf-8'
$ErrorActionPreference = 'Stop'
$script:P1QualityRoot = (Resolve-Path -LiteralPath $Suite).Path
$script:P1QualityDirectory = Join-Path $script:P1QualityRoot ('quality/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
[void][IO.Directory]::CreateDirectory($script:P1QualityDirectory)
$script:P1QualityRecords = [Collections.Generic.List[object]]::new()
$script:P1QualityInputs = [ordered]@{}
$qualityInputFiles = @((Get-Item Cargo.toml,Cargo.lock,rust-toolchain.toml))
$qualityInputFiles += Get-ChildItem crates,configs,tests/fixtures,tools/validation -Recurse -File | Where-Object { $_.Extension -in @('.rs','.toml','.json','.safetensors','.py','.ps1') }
$qualityInputFiles += Get-ChildItem tools/reference -File | Where-Object { $_.Extension -in @('.py','.toml','.lock') }
foreach ($qualityInputFile in $qualityInputFiles) {
    $qualityInputName = [IO.Path]::GetRelativePath((Get-Location).Path, $qualityInputFile.FullName).Replace('\','/')
    $script:P1QualityInputs[$qualityInputName] = (Get-FileHash -LiteralPath $qualityInputFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Invoke-QualityCommand {
    param([string]$Name, [string]$Executable, [string[]]$Arguments, [bool]$ExpectedFailure = $false)
    $qualityLog = Join-Path $script:P1QualityDirectory ($Name + '.log')
    [IO.File]::WriteAllText($qualityLog, '')
    $qualityStarted = [DateTime]::UtcNow.ToString('o')
    $qualityClock = [Diagnostics.Stopwatch]::StartNew()
    Write-Output "START quality $Name"
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $qualityLog
    $qualityExit = $LASTEXITCODE
    $qualityPassed = if ($ExpectedFailure) { $qualityExit -ne 0 } else { $qualityExit -eq 0 }
    if ($ExpectedFailure) {
        $qualityText = Get-Content -LiteralPath $qualityLog -Raw
        $qualityPassed = $qualityPassed -and $qualityText.Contains('requested_device: cuda:9') -and $qualityText.Contains('device initialization')
    }
    $script:P1QualityRecords.Add([ordered]@{
        key=$Name; executable=$Executable; arguments=$Arguments; started_at=$qualityStarted;
        ended_at=[DateTime]::UtcNow.ToString('o'); seconds=$qualityClock.Elapsed.TotalSeconds;
        exit_code=$qualityExit; expected_failure=$ExpectedFailure; passed=$qualityPassed;
        log=$qualityLog; log_sha256=(Get-FileHash -LiteralPath $qualityLog -Algorithm SHA256).Hash.ToLowerInvariant();
        rustup_auto_install=$env:RUSTUP_AUTO_INSTALL; rustup_toolchain=$env:RUSTUP_TOOLCHAIN;
        rustc=$env:RUSTC; rustdoc=$env:RUSTDOC; target=$env:CARGO_BUILD_TARGET;
        cuda_path=$env:CUDA_PATH; cl=$env:CL; rustflags=$env:RUSTFLAGS; nvcc_append_flags=$env:NVCC_APPEND_FLAGS
    })
    $qualityRecord = [ordered]@{
        passed=$qualityPassed; script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();
        records=@($script:P1QualityRecords.ToArray()); inputs=$script:P1QualityInputs
    }
    $qualityRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $script:P1QualityDirectory 'quality.json') -Encoding utf8
    Write-Output "END quality $Name exit=$qualityExit passed=$qualityPassed"
    if (-not $qualityPassed) { throw "Quality gate failed: $Name" }
}
. "$PSScriptRoot/p1_env.ps1" -Mode cpu
Invoke-QualityCommand 'cpu-fmt' $script:P1Cargo @('fmt','--all','--check')
Invoke-QualityCommand 'cpu-build' $script:P1Cargo @('build','--workspace')
Invoke-QualityCommand 'cpu-clippy' $script:P1Cargo @('clippy','--workspace','--all-targets','--','-D','warnings')
Invoke-QualityCommand 'cpu-test' $script:P1Cargo @('test','--workspace')
. "$PSScriptRoot/p1_env.ps1" -Mode cuda
Invoke-QualityCommand 'cuda-build' $script:P1Cargo @('build','--workspace','--features','cuda')
Invoke-QualityCommand 'cuda-clippy' $script:P1Cargo @('clippy','--workspace','--all-targets','--features','cuda','--','-D','warnings')
Invoke-QualityCommand 'cuda-test' $script:P1Cargo @('test','--workspace','--features','cuda')
Invoke-QualityCommand 'cuda-doctor' $script:P1Cargo @('run','-p','ferrots-cli','--features','cuda','--','doctor','--device','cuda:0','--training-smoke-test')
Invoke-QualityCommand 'cuda-invalid-device' $script:P1Cargo @('run','-p','ferrots-cli','--features','cuda','--','doctor','--device','cuda:9') $true
Invoke-QualityCommand 'python-tests' 'uv' @('run','--frozen','--project','tools/reference','python','-m','unittest','discover','-s','tools/validation/tests','-p','test_*.py')
Invoke-QualityCommand 'task-validate' 'python' @('.trellis/scripts/task.py','validate','.trellis/tasks/09-28-p1-data-train-loop')
Invoke-QualityCommand 'private-data' 'bash' @('tools/check_private_data.sh')
Invoke-QualityCommand 'diff-check' 'git' @('diff','--check')
$qualityFinal = Get-Content -LiteralPath (Join-Path $script:P1QualityDirectory 'quality.json') -Raw
$qualityFinal | Set-Content -LiteralPath (Join-Path $script:P1QualityRoot 'quality.json') -Encoding utf8
Write-Output "PASS full quality gates; records: $script:P1QualityDirectory"
