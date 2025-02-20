gmx grompp -f mdp/pull.mdp -c step7_production.gro -r step7_production.gro -p topol.top -maxwarn -1 -o pull.tpr -n index.ndx

gmx mdrun -deffnm pull -ntmpi 1 -ntomp 24 -pin on -v 

gmx sasa -f pull.trr -s pull.tpr -n index.ndx -surface 'pep'  -output '"Hydrophobic" group "pep" and charge {-0.2 to 0.2}; "Hydrophilic" group "pep" and not charge {-0.2 to 0.2}'

# python area.py area.xvg

gmx rms -s pull.tpr -f pull.trr -o rmsd.xvg <<EOF
3
3
EOF

gmx gyrate -s pull.tpr -f pull.trr -o gyrate.xvg <<EOF
3
EOF

python rmsd_gyrate.py rmsd.xvg gyrate.xvg

python merge.py rmsd.xvg gyrate.xvg merge.xvg

gmx sham -tsham 315 -nlevels 100 -f merge.xvg -ls gibbs.xpm -g gibbs.log -lsh enthalpy.xpm -lss entropy.xpm

python xpm2png.py -ip yes -f gibbs.xpm
# python xpm2png.py -ip yes -f entropy.xpm
# python xpm2png.py -ip yes -f enthalpy.xpm
# python xpm2png.py -ip yes -f prob.xpm

dit xpm2csv -f gibbs.xpm -o gibbs.csv

python gibbs_utils.py gibbs.csv merge.xvg area.xvg

rm rmsd.xvg gyrate.xvg bindex.ndx ener.xvg gibbs.log enthalpy.xpm entropy.xpm gibbs.xpm prob.xpm