#!/bin/bash
fn=$1
ofn=${fn/smartmeter001_/smartmeter001_split_}

ffmpeg -i $fn \
-map 0 -map_channel 0.0.0:0.0.0 -map_channel 0.0.1:0.0.1 \
-map 0 -map_channel 0.0.2:0.1.0 -map_channel 0.0.3:0.1.1 \
-map 0 -map_channel 0.0.4:0.2.0 -map_channel 0.0.5:0.2.1 \
-c:a wavpack \
-metadata:s:a:0 title="smartmeter001 L1" -metadata:s:a:0 CHANNELS=2 -metadata:s:a:0 CHANNEL_TAGS="v,i" \
-metadata:s:a:1 title="smartmeter001 L2" -metadata:s:a:1 CHANNELS=2 -metadata:s:a:1 CHANNEL_TAGS="v,i" \
-metadata:s:a:2 title="smartmeter001 L3" -metadata:s:a:2 CHANNELS=2 -metadata:s:a:2 CHANNEL_TAGS="v,i" \
-y $ofn