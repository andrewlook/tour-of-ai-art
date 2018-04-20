#!/bin/bash

jupyter nbconvert notebooks/a_whirlwind_tour_of_ai_art.ipynb \
    --to slides \
    --reveal-prefix reveal.js 
