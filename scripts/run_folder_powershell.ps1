param(
    [Parameter(Mandatory)]
    [string]$ImageFolder,
    [string]$Config = "config.yaml"
)
$files = @(Get-ChildItem -Path "$ImageFolder\*" -Include "*.nd2","*.czi" -File -ErrorAction SilentlyContinue)
if ($files.Count -eq 0) {
    Write-Host "No .nd2 or .czi files found in: $ImageFolder"
    exit 0
}
Write-Host "Found $($files.Count) image(s). Processing..."
$n = 0
foreach ($f in $files) {
    $n++
    Write-Host "[$n/$($files.Count)] Processing: $($f.Name)"
    pixi run python main.py --image $f.FullName --config $Config
}
Write-Host "Finished $($files.Count) image(s)."
