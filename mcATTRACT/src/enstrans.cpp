#include "state.h"
#include <cmath>
#include <iostream>

extern CartState &cartstate_get(int handle);

void calc_ensw(CartState &cs, int lig) {
  int nrens =  cs.nrens[lig];

  double **ensw = new double *[nrens];
  double **rmsds = new double *[nrens];
  int n;
  for (n = 0; n < nrens; n++) {
    ensw[n] = new double[nrens];
  }
  for (n = 0; n < nrens; n++) {
    rmsds[n] = new double[nrens];
  }

  int atomsize = cs.ieins[lig];
  if (lig > 0) atomsize -= cs.ieins[lig-1];

  for (n = 0; n < nrens; n++) {
    //    ensw[n][n] = 1.0; // 0 RMSD = 1 / (0+1.0)
    rmsds[n][n] = 0;
    double *ensd1 = cs.ensd[lig][n];
    for (int nn = n+1; nn < nrens; nn++) {
      double *ensd2 = cs.ensd[lig][nn];
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
      //      printf("lig[%d] ensw[%d][%d]=%.3f",lig,n,nn,weight);
      //      std::cout << "lig = " << lig << " ensw[ " << n << " ][ " << nn << " ] = " << weight << std::endl;
    }
  }


  cs.ensw[lig] = ensw;
}

extern "C" void enstrans_(const int &cartstatehandle, const int &lig, const int &curr, const double &rand, int &ret) {
  CartState &cs = cartstate_get(cartstatehandle);
  if (cs.ensw[lig] == NULL) calc_ensw(cs, lig);

  double **ensw = cs.ensw[lig];
  int nrens = cs.nrens[lig];
  double enswsum = 0;
  for (int n = 0; n < nrens; n++) {
    enswsum += ensw[curr-1][n];
    //    printf("lig = %d, ensw[%d][%d] = %f\n",lig, curr-1, n, ensw[curr-1][n]);
  }

  double accum = 0;
  for (int n = 0; n < nrens; n++) {
    accum += ensw[curr-1][n]/enswsum;
    if (accum > rand) {
      //printf("ENSTRANS! %d %d %.3f %.3f\n", curr, n+1, accum, rand);
      ret = n+1;
      return;
    }
  }
  //printf("ENSTRANS! %d %d %.3f %.3f\n", curr, nrens, 1.0, rand);
  ret = nrens;
  return;

}
