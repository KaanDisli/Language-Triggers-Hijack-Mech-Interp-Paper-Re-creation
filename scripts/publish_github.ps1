[CmdletBinding()]
param(
    [string]$Owner = "KaanDisli",
    [string]$Repository = "Language-Triggers-Hijack-Mech-Interp-Paper-Re-creation",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public",
    [string]$CommitMessage = "Publish learned language-trigger analysis and dashboard",
    [string]$GhPath = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $root
$gitSafeDirectory = $root.Replace("\", "/")

function Invoke-RepositoryGit {
    & git -c "safe.directory=$gitSafeDirectory" @args
}

if (-not (Test-Path -LiteralPath "docs\index.html")) {
    throw "docs/index.html is missing; render the public dashboard before publishing"
}
$dashboard = Get-Content -Raw -Encoding utf8 "docs\index.html"
foreach ($forbidden in @("C:\Users\", "C:/Users/", "file://")) {
    if ($dashboard.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        throw "docs/index.html contains a local-path marker: $forbidden"
    }
}

if (-not (Test-Path -LiteralPath ".git")) {
    Invoke-RepositoryGit init -b main
}
Invoke-RepositoryGit branch -M main
if ($LASTEXITCODE -ne 0) {
    throw "could not prepare the main branch"
}

if (-not $GhPath) {
    $installed = Get-Command gh -ErrorAction SilentlyContinue
    if ($installed) {
        $GhPath = $installed.Source
    } else {
        $portable = Get-ChildItem -Recurse -File "outputs\tools" -Filter gh.exe `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($portable) {
            $GhPath = $portable.FullName
        }
    }
}
if (-not $GhPath -or -not (Test-Path -LiteralPath $GhPath)) {
    throw "GitHub CLI was not found; install gh or pass -GhPath"
}

& $GhPath auth status *> $null
if ($LASTEXITCODE -ne 0) {
    & $GhPath auth login --hostname github.com --git-protocol https --web --clipboard
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub authentication did not complete"
    }
}

Invoke-RepositoryGit add --all
if ($LASTEXITCODE -ne 0) {
    throw "git add failed"
}
$staged = @(Invoke-RepositoryGit diff --cached --name-only --diff-filter=ACMR)
$forbiddenPaths = @(
    $staged | Where-Object {
        $_ -match '^(outputs|\.venv|\.venv-lora|build|dist)/' -or
        $_ -match '\.(safetensors|bin|pem|key)$'
    }
)
if ($forbiddenPaths) {
    throw "refusing to publish forbidden paths: $($forbiddenPaths -join ', ')"
}
foreach ($relative in $staged) {
    $item = Get-Item -LiteralPath (Join-Path $root $relative)
    if ($item.Length -gt 50MB) {
        throw "refusing to publish a file over 50 MiB: $relative"
    }
}
Invoke-RepositoryGit diff --cached --check
if ($LASTEXITCODE -ne 0) {
    throw "staged content failed git diff --check"
}

Invoke-RepositoryGit diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    Invoke-RepositoryGit commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed"
    }
}

$slug = "$Owner/$Repository"
& $GhPath repo view $slug --json url *> $null
if ($LASTEXITCODE -ne 0) {
    $visibilityFlag = "--$Visibility"
    & $GhPath repo create $slug $visibilityFlag --source . --remote origin --push `
        --description "Benign learned language-trigger proof of concept with causal and representation analysis"
    if ($LASTEXITCODE -ne 0) {
        throw "repository creation or initial push failed"
    }
} else {
    $expectedRemote = "https://github.com/$slug.git"
    $currentRemote = Invoke-RepositoryGit remote get-url origin 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-RepositoryGit remote add origin $expectedRemote
    } elseif ($currentRemote -ne $expectedRemote) {
        throw "origin points to $currentRemote instead of $expectedRemote"
    }
    Invoke-RepositoryGit push --set-upstream origin main
    if ($LASTEXITCODE -ne 0) {
        throw "push failed"
    }
}

& $GhPath api "repos/$slug/pages" *> $null
$pagesExist = $LASTEXITCODE -eq 0
$method = if ($pagesExist) { "PUT" } else { "POST" }
& $GhPath api --method $method "repos/$slug/pages" `
    -f "build_type=legacy" `
    -f "source[branch]=main" `
    -f "source[path]=/docs" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "GitHub Pages configuration failed"
}

$repositoryUrl = & $GhPath repo view $slug --json url --jq .url
$pagesUrl = "https://$($Owner.ToLowerInvariant()).github.io/$Repository/"
Write-Host "Repository: $repositoryUrl"
Write-Host "Dashboard:  $pagesUrl"
