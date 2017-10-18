#!/bin/bash

ffmpeg -i acute.png -i pic_${1}a.png -filter_complex "[1]scale=iw*4:-1[b];[0:v][b] overlay" out${1}.png
