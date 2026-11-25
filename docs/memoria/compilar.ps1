<#
.SYNOPSIS
    Compila la memoria del TFM (docs/memoria/main.tex) a PDF usando XeLaTeX y Biber.
#>
$ErrorActionPreference = "Stop"
$memoriaDir = $PSScriptRoot

$miktexBin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if ((Test-Path $miktexBin) -and ($env:PATH -notlike "*$miktexBin*")) {
    $env:PATH = "$miktexBin;$env:PATH"
}

Push-Location $memoriaDir
try {
    xelatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "Error en primera pasada de xelatex" }

    biber main
    if ($LASTEXITCODE -ne 0) { throw "Error ejecutando biber" }

    xelatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "Error en segunda pasada de xelatex" }

    xelatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "Error en tercera pasada de xelatex" }

    Write-Host "Compilacion completada: $memoriaDir\main.pdf" -ForegroundColor Green
}
finally {
    Pop-Location
}
