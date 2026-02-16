@ECHO OFF

set SOURCEDIR=source
set BUILDDIR=_build

if "%SPHINXBUILD%"=="" (
	set SPHINXBUILD=sphinx-build
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
