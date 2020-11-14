#!/usr/bin/env sh
set -eu
input=$1
ffmpeg -i $input -vf "fps=10,scale=300:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(basename -s .webm $input).gif
