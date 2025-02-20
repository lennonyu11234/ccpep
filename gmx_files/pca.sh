#!/bin/csh
#


gmx covar -s pull.tpr -f pull.trr -o eigenvalues.xvg -v eigenvectors.trr -xpma covapic.xpm <<EOF
3
3
EOF

python xpm2png.py -ip yes -f covapic.xpm

gmx anaeig -s pull.tpr -f pull.trr -v eigenvectors.trr -first 1 -last 1 -proj pc1.xvg <<EOF
3
3
EOF


gmx anaeig -s pull.tpr -f pull.trr -v eigenvectors.trr -first 2 -last 2 -proj pc2.xvg <<EOF
3
3
EOF

python merge.py pc1.xvg pc2.xvg merge_pc.xvg

gmx sham -tsham 315 -nlevels 100 -f merge_pc.xvg -ls gibbs_pc.xpm -g gibbs_pc.log -lsh enthalpy_pc.xpm -lss entropy_pc.xpm


python xpm2png.py -ip yes -f gibbs_pc.xpm

dit xpm2csv -f gibbs_pc.xpm -o gibbs_pc.csv

python pca_util.py gibbs_pc.csv merge_pc.xvg 

rm average.pdb covar.log covapic.xpm eigenvalues.xvg eigenvectors.trr bindex.ndx ener.xvg enthalpy_pc.xpm entropy_pc.xpm gibbs_pc.log gibbs_pc.xpm prob.xpm