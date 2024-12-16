
$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$root_dir = Resolve-Path -Path "$script_dir\..\..\.."

Push-Location -Path "$root_dir\submodules\hiprt"

cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=ON -DNO_ENCRYPT=ON -S . -B build
if(-not $?)
{
    Pop-Location
    exit 1
}

cmake --build build --config Release
if(-not $?)
{
    Pop-Location
    exit 1
}

cmake -DCMAKE_BUILD_TYPE=Debug -DBITCODE=ON -DNO_ENCRYPT=ON -S . -B build/debug
if(-not $?)
{
    Pop-Location
    exit 1
}

cmake --build build/debug --config Debug
if(-not $?)
{
    Pop-Location
    exit 1
}

Push-Location -Path "scripts\bitcodes"

python compile.py
Pop-Location
Pop-Location
if(-not $?)
{
    exit 1
}
