$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$root_dir = Resolve-Path -Path "$script_dir\..\..\.."

cmake --build "$root_dir\build\debug" 
if(-not $?)
{
    exit 1
}

Push-Location -Path "$root_dir\build\debug\src\app"
./hipdemo.exe
Pop-Location
