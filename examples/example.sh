#!/bin/sh

./fta_model_aual.py -au=200 -minal=-100 -maxal=-900 -outfile=examples/example_fta_au_al

# Fuller-Rowell & Evans model:
./fta_model_aual.py -fre -hp=10 
./fta_model_aual.py -fre -hp=20 
./fta_model_aual.py -fre -hp=30 
./fta_model_aual.py -fre -hp=40 
./fta_model_aual.py -fre -hp=50 
./fta_model_aual.py -fre -hp=60 
./fta_model_aual.py -fre -hp=70 
./fta_model_aual.py -fre -hp=80 
./fta_model_aual.py -fre -hp=90 
./fta_model_aual.py -fre -hp=100 
