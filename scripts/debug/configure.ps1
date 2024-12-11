$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

$rootDir = [System.IO.Path]::GetFullPath("$scriptDir/../..")

cmake -S $rootDir -B ./build/debug -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug