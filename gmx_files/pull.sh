#!/bin/bash
#


for i in {0..36} #you shoud edit the number of windows your will calculate
do 
    gmx grompp -f mdp/npt_pull.mdp -c conf$i.gro -n index.ndx -p topol.top -o um0-$i.tpr -maxwarn 99 -r conf$i.gro
	gmx mdrun -v -deffnm um0-$i -ntmpi 1 -ntomp 12 
	gmx grompp -f mdp/prd_pull.mdp -c um0-$i.gro -n index.ndx -p topol.top -o um-$i.tpr -maxwarn 99 -t um0-$i.cpt -r conf$i.gro
	gmx mdrun -deffnm um-$i -px $i-px  -pf $i-pf -ntmpi 1 -ntomp 12 -v  
done

ls um-*.tpr > tpr.dat

ls *-pf.xvg > pullf.dat

gmx wham -it tpr.dat -if pullf.dat -bsres -bins 40 -unit kJ -nBootstrap 200