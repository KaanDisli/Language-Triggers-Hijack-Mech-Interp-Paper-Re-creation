[CmdletBinding()]
param(
    [string]$Owner = "KaanDisli",
    [string]$Repository = "language-trigger-heads",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public",
    [string]$CommitMessage = "Publish learned language-trigger analysis and dashboard",
    [string]$GhPath = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $root

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
    git init -b main
}
git branch -M main
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

git add --all
if ($LASTEXITCODE -ne 0) {
    throw "git add failed"
}
$staged = @(git diff --cached --name-only --diff-filter=ACMR)
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
git diff --cached --check
if ($LASTEXITCODE -ne 0) {
    throw "staged content failed git diff --check"
}

git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m $CommitMessage
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
    $currentRemote = git remote get-url origin 2>$null
    if ($LASTEXITCODE -ne 0) {
        git remote add origin $expectedRemote
    } elseif ($currentRemote -ne $expectedRemote) {
        throw "origin points to $currentRemote instead of $expectedRemote"
    }
    git push --set-upstream origin main
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
