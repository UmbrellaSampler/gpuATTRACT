#!/bin/bash -i

###Converted from AttractEasyModel by easy2model 1.1
### Generated by ATTRACT shell script generator version 0.5

set -u -e
if [ ! -f $ATTRACTDIR/../version ] ||  [ `awk '{print ($1 < 0.3)}' $ATTRACTDIR/../version` -eq 1 ]; then
  echo 'Your ATTRACT version is too old to run this protocol. Please download and install the latest version of ATTRACT'
  echo 'ATTRACT is available at: http://www.attract.ph.tum.de/services/ATTRACT/attract.tgz'
  exit 1
fi
trap "kill -- -$BASHPID; $ATTRACTDIR/shm-clean" ERR EXIT
$ATTRACTDIR/shm-clean

rm -rf result.dat result.pdb result.lrmsd result.irmsd result.fnat >& /dev/null

#name of the run
name=demo_xylanase 

#docking parameters
params="$ATTRACTDIR/../attract.par partner1-ensemble/model-1r.pdb partner2-ensemble/model-1r.pdb --fix-receptor --ens 1 partner1-ensemble.list --ens 2 partner2-ensemble.list"
scoreparams="$ATTRACTDIR/../attract.par partner1-ensemble/model-1r.pdb partner2-ensemble/model-1r.pdb --score --fix-receptor --ens 1 partner1-ensemble.list --ens 2 partner2-ensemble.list"

#grid parameters
gridparams=" --grid 1 receptorgrid.gridheader"

#parallelization parameters
parals="--np 8 --chunks 8"
if [ 1 -eq 0 ]; then ### move and change to disable parts of the protocol

echo '**************************************************************'
echo 'Generate starting structures...'
echo '**************************************************************'
cat $ATTRACTDIR/../rotation.dat > rotation.dat
$ATTRACTDIR/translate partner1-ensemble/model-1r.pdb partner2-ensemble/model-1r.pdb > translate.dat
$ATTRACTDIR/systsearch > systsearch.dat
start=systsearch.dat

echo '**************************************************************'
echo 'ensemble search:' 
echo ' add ensemble conformations to the starting structures'
echo '**************************************************************'

echo '**************************************************************'
echo 'random ensemble conformation in ligand 1 for each starting structure'
echo '**************************************************************'
python $ATTRACTTOOLS/ensemblize.py systsearch.dat 31 1 random  > systsearch-ens1.dat

echo '**************************************************************'
echo 'random ensemble conformation in ligand 2 for each starting structure'
echo '**************************************************************'
python $ATTRACTTOOLS/ensemblize.py systsearch-ens1.dat 31 2 random  > systsearch-ens1-ens2.dat

echo '**************************************************************'
echo 'calculate receptorgrid grid'
echo '**************************************************************'
awk '{print substr($0,58,2)}' partner2-ensemble/model-1r.pdb | sort -nu > receptorgrid.alphabet
$ATTRACTDIR/make-grid-omp partner1-ensemble/model-1r.pdb $ATTRACTDIR/../attract.par 10.0 12.0 receptorgrid.grid --alphabet receptorgrid.alphabet

fi ### move to disable parts of the protocol

if [ 1 -eq 0 ]; then ### move and change to disable parts of the protocol
echo '**************************************************************'
echo 'Docking'
echo '**************************************************************'
date > TIME_CONTROL
echo '**************************************************************'
echo '1st minimization'
echo '**************************************************************'
$ATTRACTDIR/shm-grid receptorgrid.grid receptorgrid.gridheader

python $ATTRACTDIR/../protocols/attract.py systsearch-ens1-ens2.dat $params $gridparams --mc --mcmax 2000 --mcensprob 1 --mctemp 0.8 --mcscalerot 1 --mcscalecenter 0.1  --gravity 2 --ub 54.4  --rstk 0.02 $parals  --output out_$name.dat
date >> TIME_CONTROL

$ATTRACTDIR/shm-clean

fi ### move to disable parts of the protocol

if [ 1 -eq 1 ]; then ### move and change to disable parts of the protocol
echo '**************************************************************'
echo 'Final rescoring'
echo '**************************************************************'
#python $ATTRACTDIR/../protocols/attract.py out_$name.dat $scoreparams --rcut 50.0 $parals --output out_$name.score
$ATTRACTDIR/shm-grid receptorgrid.grid receptorgrid.gridheader
python $ATTRACTDIR/../protocols/attract.py out_$name.dat $scoreparams $gridparams $parals --output out_$name.grid.score
$ATTRACTDIR/shm-clean

fi ### move to disable parts of the protocol

if [ 1 -eq 0 ]; then ### move and change to disable parts of the protocol
     
echo '**************************************************************'
echo 'Merge the scores with the structures'
echo '**************************************************************'
#python $ATTRACTTOOLS/fill-energies.py out_$name.dat out_$name.score > out_$name-scored.dat

echo '**************************************************************'
echo 'Sort structures'
echo '**************************************************************'
#python $ATTRACTTOOLS/sort.py out_$name-scored.dat > out_$name-sorted.dat

echo '**************************************************************'
echo 'Remove redundant structures'
echo '**************************************************************'
#$ATTRACTDIR/deredundant out_$name-sorted.dat 2 --ens 31 31 | python $ATTRACTTOOLS/fill-deredundant.py /dev/stdin out_$name-sorted.dat > out_$name-sorted-dr.dat

echo '**************************************************************'
echo 'Soft-link the final results'
echo '**************************************************************'
#ln -s out_$name-sorted-dr.dat result.dat

echo '**************************************************************'
echo 'collect top 1 structures:'
echo '**************************************************************'
#$ATTRACTTOOLS/top out_$name-sorted-dr.dat 1 > out_$name-top1.dat
#$ATTRACTDIR/collect out_$name-top1.dat partner1-ensemble/model-1-aa.pdb partner2-ensemble/model-1-aa.pdb --ens 1 partner1-ensemble-aa.list --ens 2 partner2-ensemble-aa.list > out_$name-top1.pdb
#ln -s out_$name-top1.pdb result.pdb

echo '**************************************************************'
echo 'calculate backbone ligand RMSD'
echo '**************************************************************'
#python $ATTRACTDIR/lrmsd.py out_$name.dat partner2-ensemble/model-1-aa.pdb ligand-rmsd.pdb --ens 1 partner1-ensemble-aa.list --ens 2 partner2-ensemble-aa.list --receptor partner1-ensemble/model-1-aa.pdb > out_$name.lrmsd
#ln -s out_$name-sorted-dr.lrmsd result.lrmsd

echo '**************************************************************'
echo 'calculate backbone interface RMSD'
echo '**************************************************************'
python $ATTRACTDIR/irmsd.py out_$name.dat partner1-ensemble/model-1-heavy.pdb refe-rmsd-1.pdb partner2-ensemble/model-1-heavy.pdb refe-rmsd-2.pdb --ens 1 partner1-ensemble-aa-rmsd.list --ens 2 partner2-ensemble-aa-rmsd.list  > out_$name.irmsd
#ln -s out_$name-sorted-dr.irmsd result.irmsd

echo '**************************************************************'
echo 'calculate fraction of native contacts'
echo '**************************************************************'
python $ATTRACTDIR/fnat.py out_$name.dat 5 partner1-ensemble/model-1-heavy.pdb refe-rmsd-1.pdb partner2-ensemble/model-1-heavy.pdb refe-rmsd-2.pdb --ens 1 partner1-ensemble-aa-rmsd.list --ens 2 partner2-ensemble-aa-rmsd.list > out_$name.fnat
#ln -s out_$name-sorted-dr.fnat result.fnat

fi ### move to disable parts of the protocol

$ATTRACTDIR/shm-clean
