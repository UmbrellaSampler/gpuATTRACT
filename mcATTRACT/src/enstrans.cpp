#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include "ensembleWeightTable.h"


EnsembleWeightTable EnsembleWeightTable::globTable;


void calc_ensw(int ensembleId) {
  int nrens =  EnsembleWeightTable::globTable.getEnsembleSize(ensembleId);

  double **& ensw = EnsembleWeightTable::globTable.getEnsembleWeights(ensembleId);
  ensw = new double* [nrens];

  double rmsds[nrens][nrens];

  int n;
  for (n = 0; n < nrens; n++) {
    ensw[n] = new double[nrens];
  }

  /* new: precalulate ensemble difference table */
  int atomsize = EnsembleWeightTable::globTable.proteinSize(ensembleId, 0);
  double ensd[nrens][3*atomsize];
  {
  	double* ensd0 = ensd[0];
	int ensembleSize = EnsembleWeightTable::globTable.getEnsembleSize(ensembleId);
	int numAtoms = EnsembleWeightTable::globTable.proteinSize(ensembleId, 0);

	for (int i = 0; i < 3*numAtoms; ++i) {
		ensd0[i] = 0.0;
	}

	float* x0 = EnsembleWeightTable::globTable.proteinPosX(ensembleId, 0);
	float* y0 = EnsembleWeightTable::globTable.proteinPosY(ensembleId, 0);
	float* z0 = EnsembleWeightTable::globTable.proteinPosZ(ensembleId, 0);
	for (int n = 1; n < ensembleSize; ++n) {
		double* ensd1 = ensd[n];
		float* x1 = EnsembleWeightTable::globTable.proteinPosX(ensembleId, n);
		float* y1 = EnsembleWeightTable::globTable.proteinPosY(ensembleId, n);
		float* z1 = EnsembleWeightTable::globTable.proteinPosZ(ensembleId, n);
		for (int i = 0; i < numAtoms; ++i) {
			ensd1[i*3 + 0] = x0[i] - x1[i];
			ensd1[i*3 + 1] = y0[i] - y1[i];
			ensd1[i*3 + 2] = z0[i] - z1[i];
		}
	}
  }

  for (n = 0; n < nrens; n++) {
    //    ensw[n][n] = 1.0; // 0 RMSD = 1 / (0+1.0)
    rmsds[n][n] = 0;
    double *ensd1 = ensd[n];
    for (int nn = n+1; nn < nrens; nn++) {
      double *ensd2 = ensd[nn];
      double sd = 0;
      for (int i = 0; i < atomsize; i++) {
        double dx = ensd1[3*i]-ensd2[3*i];
        double dy = ensd1[3*i+1]-ensd2[3*i+1];
        double dz = ensd1[3*i+2]-ensd2[3*i+2];
        double dsq = dx*dx+dy*dy+dz*dz;
        sd += dsq;
      }
      double msd = sd/atomsize;
      double rmsd = sqrt(msd);
      rmsds[n][nn] = rmsd;
      rmsds[nn][n] = rmsd;
      //      printf("ens RMSD %d - %d: %.3f\n", n+1,nn+1, rmsd);
//       double weight = 1.0/(rmsd+1.0);
//       ensw[n][nn] = weight;
//       ensw[nn][n] = weight;
    }
  }

  double cumu;
  for ( n=0; n<nrens; n++ ){
    cumu = 0.0;
    for ( int nn = 0; nn < nrens; nn++ ) {
      if ( nn != n ) {
	cumu += 1.0/(rmsds[n][nn]+1); // +1 to avoid singularity

      }
    }
    rmsds[n][n] = (nrens-1)/cumu-1; // such that it has a rate of 1/n to stay in the current model
  }

  for ( n = 0; n < nrens; n++ ) {
    for ( int nn = n; nn < nrens; nn++ ) {
      double weight = 1./(rmsds[n][nn]+1);
      ensw[n][nn] = weight; //1.0/(rmsds[n][nn]+0.0001);
      ensw[nn][n] = weight;
//            printf("lig[%d] ensw[%d][%d]=%.3f",ensembleId,n,nn,weight);
//            std::cout << "lig = " << ensembleId << " ensw[ " << n << " ][ " << nn << " ] = " << weight << std::endl;
    }
  }

}

extern "C" void enstrans_(const int &cartstatehandle, const int &lig, const int &curr, const double &rand, int &ret) {

  int nrens =  EnsembleWeightTable::globTable.getEnsembleSize(lig);
  double** ensw = EnsembleWeightTable::globTable.getEnsembleWeights(lig);

  double enswsum = 0;
  for (int n = 0; n < nrens; n++) {
    enswsum += ensw[curr-1][n];
//        printf("lig = %d, ensw[%d][%d] = %f\n",lig, curr-1, n, ensw[curr-1][n]);
  }

  double accum = 0;
  for (int n = 0; n < nrens; n++) {
    accum += ensw[curr-1][n]/enswsum;
    if (accum > rand) {
//      printf("ENSTRANS1! %d %d %.3f %.3f\n", curr, n+1, accum, rand);
      ret = n+1;
      return;
    }
  }
//  printf("ENSTRANS2! %d %d %.3f %.3f\n", curr, nrens, 1.0, rand);
  ret = nrens;
  return;

}
