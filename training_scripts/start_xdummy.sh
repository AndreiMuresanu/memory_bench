#!/bin/bash

# Create fake framebuffer device
sudo modprobe v4l2loopback

# Start Xdummy on display :1
Xorg :1 -config ./xorg.conf.dummy -noreset -logfile ./tmp/xorg.log &
sleep 2

# Export DISPLAY
export DISPLAY=:1

# Now run your Unity script
vglrun -d :1 python3 ml_agents_input_test.py