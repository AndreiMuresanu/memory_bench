#!/bin/bash

xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
MultiTask_noMem.yaml \
--run-id multitask_no-mem_0 --seed=0 --base-port=5005 \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/MultiTask/linux/pixel_input/gamefile.x86" \
&& 
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
MultiTask_noMem.yaml \
--run-id multitask_no-mem_1 --seed=1 --base-port=5006 \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/MultiTask/linux/pixel_input/gamefile.x86" \
&& 
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
MultiTask_noMem.yaml \
--run-id multitask_no-mem_2 --seed=2 --base-port=5007 \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/MultiTask/linux/pixel_input/gamefile.x86" \
&& 
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
MultiTask_noMem.yaml \
--run-id multitask_no-mem_3 --seed=3 --base-port=5008 \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/MultiTask/linux/pixel_input/gamefile.x86" \
&& 
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
MultiTask_noMem.yaml \
--run-id multitask_no-mem_4 --seed=4 --base-port=5009 \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/MultiTask/linux/pixel_input/gamefile.x86"
