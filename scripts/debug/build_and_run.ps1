$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$rootDir = Resolve-Path -Path "$scriptDir\..\.."

try {
    cmake --build "$rootDir\build\debug" 
} catch {
    exit 1
}

Push-Location -Path "$rootDir\build\debug\src\app"

try {
    ./hipdemo.exe
} finally {
    Pop-Location
}