#!/bin/bash

echo "This script is used to faster generate the initial model of umbrella sampling and writed by Yunsen Zhang at 24/03/2023"

#please announce the variable you used in your system like: 'bash pipeline.sh molecule.gro solute.pdb 6.38 6.38 20 5'
molecule="DOPC.gro"  #membrane name
solute="pep.pdb"  #molecule you want to add to membrane
box_size_x="6" #box size: x axis
box_size_y="6" #box size: y axis
box_size_z="20" #box size: z axis
dist="2"

#box settings
center_x="3"
center_y="3"
center_z="10"
pro_z="14"
pot_z="15"
mem_low="8"
mem_high="12"

#box generating
gmx editconf -f $molecule -c -o step1.gro -box $box_size_x $box_size_y $box_size_z -center $center_x $center_y $center_z
gmx genconf -f $solute -o solute.pdb
gmx editconf -f solute.pdb -o solute_c.gro -box $box_size_x $box_size_y $box_size_z -center $center_x $center_y $pro_z
sed '1,2d' solute_c.gro > solute_temp.gro
sed '1,2d;$d' step1.gro > step_temp.gro
A_lines=$(cat step_temp.gro | wc -l)
B_lines=$(cat solute_temp.gro | wc -l)
atom_number=$(expr $A_lines + $B_lines - 1)
echo -e "MOL\n$atom_number" > part.gro
cat part.gro step_temp.gro solute_temp.gro > whole.gro

#add virtual site
gmx genconf -f POT.pdb -o K.gro
gmx editconf -f K.gro -o K_box.gro -box $box_size_x $box_size_y $box_size_z -center $center_x $center_y $pot_z
gmx solvate -cp whole.gro -cs K_box.gro -o mol.gro
gmx solvate -cp mol.gro -cs spc216.gro -o solv.gro

#solvate the box
gmx make_ndx -f solv.gro -o SOL.ndx <<EOF
! r SOL
q
EOF
gmx editconf -f solv.gro -n SOL.ndx -o WAT.gro <<EOF 
SOL
EOF
gmx select -f WAT.gro -s WAT.gro -select "resname SOL and res_com z > 8 and res_com z < 12" -on WAT_PART.ndx
gmx make_ndx -f WAT.gro -o NO_WAT.ndx -n WAT_PART.ndx <<EOF
! 0
q
EOF
gmx editconf -f WAT.gro -n NO_WAT.ndx -o WAT_PART.gro <<EOF
1
EOF
gmx editconf -f solv.gro -n SOL.ndx -o SOLU.gro <<EOF
!SOL
EOF

#generate the prepared .gro
gmx solvate -cp SOLU.gro -cs WAT_PART.gro -o step5_ready.gro

#remove redundancy

rm SOLU.gro WAT_PART.gro WAT.gro WAT_PART.ndx SOL.ndx solv.gro mol.gro whole.gro step1.gro step_temp.gro solute_temp.gro solut_c.gro solute.pdb part.gro NO_WAT.ndx solute_c.gro

gmx make_ndx -f step5_ready.gro -o index.ndx<<EOF
q
EOF