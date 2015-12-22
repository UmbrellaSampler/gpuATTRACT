/*
 * ensembeWeightTable.h
 *
 *  Created on: Dec 22, 2015
 *      Author: uwe
 */

#ifndef ENSEMBEWEIGHTTABLE_H_
#define ENSEMBEWEIGHTTABLE_H_

#include <vector>
#include "as/Protein.h"

constexpr int numBindingPartners = 2; // For two body docking

void calc_ensw(int ensembleId);

class EnsembleWeightTable {
public:
	EnsembleWeightTable() {}

	~EnsembleWeightTable()
	{
		for (int i = 0; i < numBindingPartners; ++i) {
			for (int n = 0; n < ensembleSize[i]; n++) {
				delete[] ensembleWeights[i][n];
			}
			delete[] ensembleWeights[i];
		}
	}

	void init() {
		assert(ensembleProteins.size() == numBindingPartners);
		for (int i = 0; i < numBindingPartners; ++i) {
			assert(ensembleProteins[i].size() > 0);
			calc_ensw(i);
		}
	}

	// needs to called before init()
	void setEnsembleProteins(std::vector<std::vector<as::Protein*>>& ensProts) {
		ensembleProteins = ensProts;
	}

	double**& getEnsembleWeights(int id) {
		return ensembleWeights[id];
	}

	int getEnsembleSize(int id) const {
		return ensembleProteins[id].size();
	}

	int proteinSize(int ensembleId, int proteinId) const {
		return ensembleProteins[ensembleId][proteinId]->numAtoms();
	}

	float* proteinPosX(int ensembleId, int proteinId) const {
		return ensembleProteins[ensembleId][proteinId]->xPos();
	}

	float* proteinPosY(int ensembleId, int proteinId) const {
		return ensembleProteins[ensembleId][proteinId]->yPos();
	}

	float* proteinPosZ(int ensembleId, int proteinId) const {
		return ensembleProteins[ensembleId][proteinId]->zPos();
	}

	static EnsembleWeightTable globTable;

private:
	double** ensembleWeights[numBindingPartners];
	int ensembleSize[numBindingPartners];

	std::vector<std::vector<as::Protein*>> ensembleProteins;

};



#endif /* ENSEMBEWEIGHTTABLE_H_ */
