#!/bin/bash


TOTAL_TIME=2500

EXTRACT_EVERY=100

TRR_FILE="pull.trr"

TPR_FILE="pull.tpr"
select_group=0
BASE_NAME="conf"

NUM_FRAMES=$(( (TOTAL_TIME / EXTRACT_EVERY) + 1 ))

count=0

for (( i=0; i<NUM_FRAMES; i++ ))
do
    CURRENT_TIME=$(( i * EXTRACT_EVERY ))
     INPUT_STRING="0"
    
        gmx trjconv -f "$TRR_FILE" -s "$TPR_FILE" -o "${BASE_NAME}${i}.gro" -pbc mol -center -dt "$EXTRACT_EVERY" -b "$CURRENT_TIME" -e "$CURRENT_TIME" <<'EOF'
0
0
EOF
    
    ((count++))
done

rm conf0.gro conf1.gro conf2.gro conf3.gro conf4.gro conf6.gro conf7.gro conf8.gro conf9.gro conf11.gro conf12.gro conf13.gro conf14.gro 
rm conf25.gro conf24.gro conf23.gro conf22.gro conf21.gro conf20.gro conf19.gro conf15.gro conf17.gro conf16.gro

mv conf5.gro init.gro
mv conf10.gro touch.gro
mv conf18.gro internalise.gro