$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$root_dir = Resolve-Path -Path "$script_dir\..\..\.."

cmake --build "$root_dir\build\debug" 