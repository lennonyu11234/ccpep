#!/bin/bash
#

gmx rms -s init_2.tpr -f init_2.xtc -o init_rmsd.xvg <<EOF
3
3
EOF

gmx gyrate -s init_2.tpr -f init_2.xtc -o init_gyrate.xvg <<EOF
3
EOF

gmx rms -s touch_2.tpr -f touch_2.xtc -o touch_rmsd.xvg <<EOF
3
3
EOF

gmx gyrate -s touch_2.tpr -f touch_2.xtc -o touch_gyrate.xvg <<EOF
3
EOF

gmx rms -s internalise_2.tpr -f internalise_2.xtc -o internalise_rmsd.xvg <<EOF
3
3
EOF

gmx gyrate -s internalise_2.tpr -f internalise_2.xtc -o internalise_gyrate.xvg <<EOF
3
EOF

python merge.py touch_rmsd.xvg touch_gyrate.xvg touch_merge.xvg

python gibbs_utils.py gibbs.csv init_merge.xvg init_area.xvg

python gibbs_utils.py gibbs.csv touch_merge.xvg touch_area.xvg

python gibbs_utils.py gibbs.csv internalise_merge.xvg internalise_area.xvg
