#!/bin/sh
# Build GITM's Fortran FTA model, run it and fta_model_aual.py at the AU, AL
# in points.txt, and compare the two.
#
# Usage: validate/run.sh <path to an Electrodynamics checkout>

if [ -z "$1" ]; then
    echo "Usage: $0 <path to an Electrodynamics checkout>"
    exit 1
fi

ED=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# GITM builds with 8-byte reals
FFLAGS="-fdefault-real-8 -fdefault-double-8 -O0"

cd "$WORK" || exit 1
gfortran $FFLAGS -c "$ED/src/ModCharSize.f90" "$ED/src/ModErrors.f90" \
    "$ED/src/ModFtaModel.f90" "$HERE/driver.f90" || exit 1
gfortran -o driver driver.o ModFtaModel.o ModErrors.o ModCharSize.o || exit 1

cp "$HERE/points.txt" .
./driver "$ED/data/ext/FTA/" || exit 1
$PYTHON "$HERE/compare.py"
