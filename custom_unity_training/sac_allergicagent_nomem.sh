#!/bin/bash

xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
SAC_AllergicAgent.yaml \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/gamefile.x86" \
--run-id sac_allergicagent_nomem_final_0 --seed=0 --base-port=5005 \
&& \
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
SAC_AllergicAgent.yaml \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/gamefile.x86" \
--run-id sac_allergicagent_nomem_final_1 --seed=1 --base-port=5006 \
&& \
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
SAC_AllergicAgent.yaml \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/gamefile.x86" \
--run-id sac_allergicagent_nomem_final_2 --seed=2 --base-port=5007 \
&& \
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
SAC_AllergicAgent.yaml \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/gamefile.x86" \
--run-id sac_allergicagent_nomem_final_3 --seed=3 --base-port=5008 \
&& \
xvfb-run -s "-screen 0 1400x900x24" -a mlagents-learn \
SAC_AllergicAgent.yaml \
--results-dir /h/andrei/memory_bench/custom_unity_training/final_results \
--force --quality-level=1 --time-scale=100 \
--env="/h/andrei/memory_bench/unity_projects/memory_palace_2/Builds/AllergicAgent/linux/pixel_input/gamefile.x86" \
--run-id sac_allergicagent_nomem_final_4 --seed=4 --base-port=5009 \
