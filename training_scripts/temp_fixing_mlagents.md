
# check existing screens
ps aux | grep X

# kill screen
pkill Xvfb

# starting an xvfb screen

module load virtualgl
Xvfb :1 -screen 0 1920x1080x16 &
export DISPLAY=:1
vglrun -d :1 python3 ml_agents_input_test.py