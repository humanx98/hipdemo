$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root_dir = Resolve-Path -Path "$script_dir\..\..\.."

cmake -S $root_dir -B ./build/debug -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug