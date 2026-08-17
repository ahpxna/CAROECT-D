#!/bin/bash
export MV_HAL_PLUGIN_PATH="$HOME/ArenaSDK_Linux_x64/Metavision/lib/metavision/hal/plugins:$HOME/ArenaSDK_Linux_x64/Metavision/hal_plugin"
export LD_LIBRARY_PATH="$HOME/ArenaSDK_Linux_x64/Metavision/lib:${LD_LIBRARY_PATH:-}"
echo ": RECORD (Arena SDK/C++)"
