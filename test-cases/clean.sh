#rm !(receptor.pdb|ligand.pdb|receptor-rmsd.pdb|ligand-rmsd.pdb|capri.py|*.sh)
find . ! -regex ".*/\(receptor.pdb\|ligand.pdb\|receptor-rmsd.pdb\|ligand-rmsd.pdb\|capri.py\|.*.sh\)" -delete
