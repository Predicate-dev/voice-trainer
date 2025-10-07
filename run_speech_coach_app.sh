#!/bin/zsh
# Launch Speech Coach App with correct Numba threading layer
export NUMBA_THREADING_LAYER=tbb
python speech_coach_app.py "$@"
