#!/bin/bash -l

# OmniGen2
nohup scripts/run_omnigen2.sh > logs/run_omnigen2.log 2>&1 &

# BAGEL
nohup scripts/run_bagel.sh > logs/run_bagel.log 2>&1 &

# UNO
nohup scripts/run_uno.sh > logs/run_uno.log 2>&1 &

# OmniGen
nohup scripts/run_omnigen.sh > logs/run_omnigen.log 2>&1 &

# Emu2
nohup scripts/run_emu2.sh > logs/run_emu2.log 2>&1 &