#!/bin/bash
START_DIR=$(pwd)

mkdir ~/.mujoco
cd ~/.mujoco && wget https://www.roboti.us/download/mjpro150_linux.zip && wget https://www.roboti.us/file/mjkey.txt && unzip mjpro150_linux.zip 