import pickle, argparse
import numpy as np
from hmmlearn import hmm
import sciL3PythonUtils as spu
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches

BIN_FACTOR = 3 # 1
KSEG = 3
EPOCH = 20

def plotPatInline(vCounts, hCounts, vSites, hSites, chrSz, cellName, chrCent, BIN_SIZE, BIN_SCALE, VAR_PER_BIN, CHR_LIST,
                  lib, states, cbl, hetState=None, zTruth=None):
        """Visualize the delta between observed heterozygosity vs MAF at shared sites

           Params:
                refC = reference allele counts (DICT key:chrI, val:list of ints)
                altC = alternate allele counts (DICT key:chrI, val:list of ints)
                refS = reference allele sites (DICT key:chrI, val:list of ints)
                altS = reference allele sites (DICT key:chrI, val:list of ints)
                BIN_SIZE = # of bp to use for binning variants + haplotypes (int)
                BIN_SCALE = The nearest value to round to (ie, 100Kbp) (int)
                VAR_PER_BIN = The average # of variants + haplotypes to target per bin (int)
        """
        figure_width = 8
        figure_height = 4
        plt.figure(figsize=(figure_width, figure_height))
        panel1_width = 7/figure_width
        panel1_height = 1/figure_height # was 2.25
        panel1=plt.axes([380/(figure_width*600), 1-1000/(figure_height*600), panel1_width, panel1_height])

        panel2_width = 7/figure_width
        panel2_height = 1/figure_height # was 2.25
        panel2=plt.axes([380/(figure_width*600), 1-1850/(figure_height*600), panel1_width, panel1_height])

        chrPlotObjects = [None for c in CHR_LIST]
        offset = int()
        for ci,chrom in enumerate(CHR_LIST):
                print(chrom)
                chrPlotObjects[ci] = spu.binChromosome(vCounts[chrom], hCounts[chrom], vSites[chrom], hSites[chrom], chrSz[chrom], offset,
                                                       BIN_SIZE, BIN_SCALE, VAR_PER_BIN)
                offset += chrSz[chrom]

        offset = int()
        centOffset = int()
        bpChrs = None
        rChrs = None
        for ci,c in enumerate(CHR_LIST):
                X_bins_var = chrPlotObjects[ci][0]
                X_bins_hap = chrPlotObjects[ci][1]
                Y_variants = chrPlotObjects[ci][2]
                Y_haplotypes = chrPlotObjects[ci][3]

                offset += chrSz[c]

                print(c, Y_variants[0:20], Y_haplotypes[0:20])
                panel1.scatter(X_bins_var, Y_variants, color='black', marker='o', s=16, alpha=0.1) # s=1.707
                panel2.scatter(X_bins_hap, Y_haplotypes, color='black', marker='o', s=16, alpha=0.1) # s=1.707
                panel1.add_patch(mplpatches.Rectangle([centOffset+chrCent[c][0], -0.02], chrCent[c][1]-chrCent[c][0], 1.04,
                                 facecolor='gray', alpha=0.3))
                panel2.add_patch(mplpatches.Rectangle([centOffset+chrCent[c][0], -0.02], chrCent[c][1]-chrCent[c][0], 1.04,
                                 facecolor='gray', alpha=0.3))
                centOffset += chrSz[c]
                # This is the chr divider line
                panel1.plot([offset, offset], [-0.02, 1.02], linewidth=1, color='black', markersize=0, alpha=0.7)
                panel2.plot([offset, offset], [-0.02, 1.02], linewidth=1, color='black', markersize=0, alpha=0.7)
                for bi,b in enumerate(cbl[ci]):
                        try:
                                nextb = cbl[ci][bi+1]
                        except:
                                nextb = chrSz[c]
                        if hetState == None:
                                #if states[ci][bi] == 0 or states[ci][bi] == 4:
                                if states[ci][bi] == 0 or states[ci][bi] == 2:
                                        # print("LOH higlight at", c, b, nextb)
                                        panel1.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='red', alpha=0.2))
                                        panel2.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='red', alpha=0.2))
                                if states[ci][bi] == 1 or states[ci][bi] == 3:
                                        # print("LOH higlight at", c, b, nextb)
                               	        panel1.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='blue', alpha=0.2))
                                        panel2.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='blue', alpha=0.2))
                        else:
                                if zTruth[ci][bi] == hetState[0] or zTruth[ci][bi] == hetState[1]:
	                                panel1.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='yellow', alpha=0.2))
                                if zTruth and if states[ci][bi] == hetState[0] or states[ci][bi] == hetState[1]:
        	                        panel2.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='red', alpha=0.2))
                                if not zTruth and if states[ci][bi] == hetState[0] or states[ci][bi] == hetState[1]:
	                                panel1.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='red', alpha=0.2))
        	                        panel2.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='red', alpha=0.2))
                                if states[ci][bi] == hetState[2] or states[ci][bi] == hetState[3]:
	                                panel1.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='blue', alpha=0.2))
        	                        panel2.add_patch(mplpatches.Rectangle([offset-chrSz[c]+b, -0.02], nextb-b, 1.04, facecolor='blue', alpha=0.2))

        panel1.set_title('Patski {0} LOH Scan'.format(cellName))
        panel1.set_ylim(0,1) # Was 0-10
        panel2.set_ylim(0,1) # Was 0-10
        panel1.set_xlim(0,offset)
        panel2.set_xlim(0,offset)

        # xtLabels = ['{0}:1-{1}'.format(c,chrSz[c]) for c in CHR_LIST]
        xtLabels = [c[3:] for c in CHR_LIST]
        #xtList = [int(0)]
        xtList = list()
        offset = int()
        for c in CHR_LIST:
                #xtList.append(chrSz[c]+xtList[-1])
                xtList.append(0.5*chrSz[c]+offset)
                offset += chrSz[c]
        panel1.set_xticks(xtList)
        panel2.set_xticks(xtList)
        panel1.set_yticks([-0.02, 0.5, 1.02])
        panel2.set_yticks([-0.02, 0.5, 1.02])
        panel1.set_xticklabels(xtLabels, fontsize=9)
        panel2.set_xticklabels(xtLabels, fontsize=9)
        panel2.set_xlabel("Genome coordinate -- Chr:Pos(Mbp)")
        panel1.set_ylabel("REF/(REF+ALT)", fontsize=9)
        panel2.set_ylabel("HAP1/(HAP1+HAP2)", fontsize=9)
        panel1.set_yticklabels([0, 0.5, 1], fontsize=8)
        panel2.set_yticklabels([0, 0.5, 1], fontsize=8)

        plt.savefig('multinomial/{0}_{1}_HmmLohPlot_ChrPainting_3StateMultinomial_{2}seg.png'.format(cellName,lib,KSEG), dpi=600)

def runHmm(cpl, hSites, chrList, args):
	"""Setup the multinomial HMM matrices and model"""

	bestModel = None
	bestLL = float('-inf')
	xList = [np.array(cpl[0][i]).reshape((len(cpl[0][i]), 1)) for i in range(len(cpl[0]))]
	X = np.concatenate(xList)
	lengths = [len(x) for x in xList]
	nSnp = sum(lengths)
	tProb = 1/(nSnp/20)

	# Optionally load a saved model
	# with open("patskiHmm.pickle", "rb") as file:
	#	bestModel = pickle.load(file)

	for epoch in range(EPOCH):
		model = hmm.MultinomialHMM(n_components=3, n_iter=10, verbose=True, params="se", init_params="se", tol=0.1, smax=KSEG)

		# For the case of fixed transition matrix
		t = np.array([[1-tProb, 0.8*tProb, 0.2*tProb], 
			      [0.5*tProb, 1-tProb, 0.5*tProb],
			      [0.2*tProb, 0.8*tProb, 1-tProb]])
		model.transmat_ = t

		model.fit(X, lengths)
		try:
			print("Stationary Distribution:", model.get_stationary_distribution())
			print("Start params:\n", model.startprob_)
			print("Transition params:\n", model.transmat_)
			print("Emission params:\n", model.emissionprob_)
		except:
			continue

		score = model.score(X, lengths)

		# Handle bad params - ValueError: rows of transmat_ must sum to 1.0 (got [1. 1. 0.])
		try:
			if score >= bestLL:
			#if model.score(X) >= bestLL:
				bestLL = score
				bestModel = model
			print("Model score", score)
			print("***BEST-LL", bestLL)
		except:
			print("Error in generating score (ie, posterior / likelihood)")
			continue

	# Optionally save the best model to file
	# with open("patskiHmm.pickle", "wb") as file:
	#	pickle.dump(bestModel, file)

	print("Evaluate best model")
	try:
		print("Stationary Distribution:", bestModel.get_stationary_distribution())
		print("Start params:\n", bestModel.startprob_)
		print("Transition params:\n" , bestModel.transmat_)
		print("Emission params:\n" , bestModel.emissionprob_)
	except:
		pass

	# Handle bad initializations: ValueError("startprob_ must sum to 1.0 (got {:.4f})"
	try:
		states = bestModel.decode(X, lengths)
		print("Likelihood of the hidden path:", states[0])
		print(np.shape(states), np.shape(states[1]), len(states[1]))
	except:
		print("Decoding error!")
		pass

	print("Best LL", bestLL)

	offset = int()
	resultList = list()
	for i in range(len(xList)):
		print("{0}".format(chrList[i]),
		      "State 0 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(0)),
		      "State 1 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(1)),
		      "State 2 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(2)),
		      "State 3 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(3)),
		      "State 4 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(4)))
		resultList.append(states[1][offset:offset+lengths[i]].tolist())
		offset += lengths[i]

	stateFreqs = bestModel.emissionprob_[:,0]*bestModel.emissionprob_[:,1]
	hetState = np.argsort(stateFreqs).tolist()
	return resultList, hetState

def runGHmm(cpl, cellBinList, chrList, args):
	"""Setup the Gaussian HMM matrices and model"""
	bestModel = None
	bestLL = float('-inf')
	xList = [np.array(cpl[0][i]).reshape((len(cpl[0][i]), 1)) for i in range(len(cpl[0]))]
	X = np.concatenate(xList)
	lengths = [len(x) for x in xList]

	# Optionally load a saved model
	# with open("patskiGHmm.pickle", "rb") as file:
	#	bestModel = pickle.load(file)

	print("--------\nRun Model\n--------\n")
	for epoch in range(EPOCH):
		model = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="mcst", params="mcst", verbose = True, tol=0.01, smax=None)

		# Optionally provide fixed mean + covar matrices for 3 state model
		model.means_ = np.array([0.025, 0.5, 0.975]).reshape((3,1))
		model.covars_ = np.array([0.0001, 0.03, 0.0001]).reshape((3,1))

		model.fit(X, lengths)
		try:
			score = model.score(X, lengths)		
			if score > bestLL:
				bestLL = score
				bestModel = model
			print("Model score", score)
			print("***BEST-LL", bestLL)
		except:
			print("Something wrong with the posterior probs or parameters...")
			continue

		try:
			print("Stationary Distribution:", model.get_stationary_distribution())
			print("Start params:\n", model.startprob_)
			print("Transition params:\n", model.transmat_)
			print("Mean params", model.means_)
			print("Covars params", model.covars_)
			# Gaussian model has no emission probs...
			# print("Emission params:\n", model.emissionprob_)
		except:
			continue

	# Optionally save the best model to file
	# with open("patskiGHmm.pickle", "wb") as file:
	#	pickle.dump(bestModel, file)

	#print(dir(bestModel))
	#print(np.shape(X), np.shape(np.transpose(X)))

	print("Evaluate best model")
	try:
		print("Stationary Distribution:", bestModel.get_stationary_distribution())
		print("Start params:\n", bestModel.startprob_)
		print("Transition params:\n" , bestModel.transmat_)
		print("Emission params:\n" , bestModel.emissionprob_)
	except:
		pass

	#bestModel.fit(X)
	#states = bestModel.predict(X)
	
	# Handle bad initializations: ValueError("startprob_ must sum to 1.0 (got {:.4f})"
	states = bestModel.decode(X)
	try:
		states = bestModel.decode(X, lengths)
		print("Likelihood of the hidden path:", states[0])
		print(np.shape(states), np.shape(states[1]), len(states[1]))
	except:
		print("Decoding error!")
		pass

	print("Best LL", bestLL)

	resultList = list()
	offset = int()
	for i in range(len(xList)):
		print("{0}".format(chrList[i]),
		      "State 0 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(0)),
		      "State 1 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(1)),
		      "State 2 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(2)),
		      "State 3 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(3)),
		      "State 4 sites: {0}".format(list(states[1][offset:offset+lengths[i]]).count(4)))
		resultList.append(states[1][offset:offset+lengths[i]].tolist())
		offset += lengths[i]

	stateFreqs = bestModel.means_[:,0]
	hetState = np.argsort(stateFreqs).tolist()
	return resultList, hetState

def simHmm(chrList, chrLenDict, cov):
	"""Simulate haplotype data using a known multinomial HMM"""
	model = hmm.MultinomialHMM(n_components=3, n_iter=10, verbose=True, params="st", init_params="st", tol=0.1, smax=KSEG)

	# emissionprob_ must have shape (n_components, n_features)
	e = np.array([[0.95, 0.05],
		      [0.5, 0.5],
		      [0.05, 0.95]])
	model.emissionprob_ = e

	t = np.array([[0.999995, 0.000001, 0.000004],
		      [0.0000025, 0.999995, 0.0000025], 
		      [0.000001, 0.000004, 0.999995]])
	model.transmat_ = t

	s = np.array([0.05, 0.9, 0.05])
	model.startprob_ = s

	sampleList = list()
	for i in range(len(chrList)):
		print("**Simulate sample", i)
		sampleX, sampleZ = model.sample(cov*chrLenDict[chrList[i]]//int(1e6))
		print("Sample info:", np.shape(sampleX), np.shape(sampleZ))
		print("SampleX", np.transpose(sampleX))
		print("SampleZ", sampleZ)
		sampleList.append((sampleX, sampleZ))

	with open("SimulatedSampleList{0}ReadsPerMb.pickle".format(COV), "wb") as file:
		pickle.dump(sampleList, file)


def formatSimulatedData(chrList, args, chrLenDict):
	"""Convert the simulated haplotype data back into sciL3 formatted haplotypes for plotting"""

	invHapMap = {0: (1, 0), 1: (0, 1), 2: (1, 1)}
	masterDict = dict()
	zTruth = list()
	masterDict[args.cellName] = [dict(), dict()]
	#with open("SimulatedSampleList.pickle", "rb") as file:
	with open("SimulatedSampleList{0}ReadsPerMb.pickle".format(COV), "rb") as file:
		sortedSimData = pickle.load(file)
		print("After X swap", [len(x[1]) for x in sortedSimData])
		for ci,c in enumerate(chrList):
			masterDict[args.cellName][0][c] = [invHapMap[symbol] for emit in sortedSimData[ci][0] for symbol in emit.flatten()]
			masterDict[args.cellName][1][c] = sorted(np.random.randint(1, chrLenDict[c], len(sortedSimData[ci][1])))
			print("Length of {0} sites".format(c), len(masterDict[args.cellName][1][c]), masterDict[args.cellName][1][c][1:10])
			zTruth.append([z for z in sortedSimData[ci][1]])
			print("Length of {0} zTruth".format(c), len(zTruth[ci]))
	return masterDict, zTruth

def main():
	parser = argparse.ArgumentParser(description='Visualize single cell genotypes and haplotypes')
	parser.add_argument('cellName', metavar='cellName', type=str, nargs='?', help='Name of the cell type yixxx_barcode1.barcode2')
	parser.add_argument('library', metavar='library', choices=['bj', 'patski'], type=str, 
			    help='python3 scanForLoh.py yiAAA_BBBBBB.CCCCCCCCCCCCCC')
	parser.add_argument('--more', dest='more', action='store_true', help='Run HMM on a list of cells instead of just 1 provided')
	parser.add_argument('--gaussian', dest='gaussian', action='store_true', help='Use Gaussian model instead of multinomial')
	parser.add_argument('--generate', dest='generate', action='store_true', help='Use a fixed parameter HMM to generate data')
	parser.add_argument('--simulate', dest='simulate', action='store_true', help='Open the generated data and learn the model')
	args = parser.parse_args()

	BIN_SIZE = 100000
	BIN_SCALE = 0 # 3 Average the bin by another 50Kbp
	if args.library == 'patski':
		chrList = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
			   'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX']
		#chrList = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8']
		dataFile = '../tools/newPickleData2/{0}_{1}MasterCellAlleleHapData.pickle'.format(args.cellName, args.library)
		CHR_LENGTHS = '../mm10_chrom_sizes.txt'
		CENTROMERES = '../mm10.centromeres.bed'
		CELL_NAMES_FILE = '../tools/PatskiCellsQCSortedUnkwFilter20210930NewSummary.txt'
		VAR_PER_BIN = 20 # 50 # 20 - 40 depending on coverage
	elif args.library == 'bj':
		chrList = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
			   'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
			   'chr20', 'chr21', 'chr22']
		dataFile = '../tools/bjPickleData/{0}_{1}MasterCellAlleleHapData.pickle'.format(args.cellName, args.library)
	
	chrLenDict = dict()
	centDict = dict()
	cellCovDict = dict()
	with open(CHR_LENGTHS) as clf, open(CENTROMERES) as centf, open(CELL_NAMES_FILE) as cf:
		spu.buildChromosomeLengthDict(clf, chrLenDict)
		spu.buildCentromereDict(centf, centDict)
		spu.buildCoverageDict(cf, cellCovDict)

	cellHaps = list()
	cellSites = list()
	if not args.more:
		TEST_CELLS = [args.cellName]
	if args.simulate:
		masterCellDataDict, zTruth = formatSimulatedData(chrList, args, chrLenDict)
		vCounts = masterCellDataDict[args.cellName][0]
		vSites = masterCellDataDict[args.cellName][1]
		cellHaps.append(masterCellDataDict[args.cellName][0])
		cellSites.append(masterCellDataDict[args.cellName][1])
	elif not args.generate:
		for cell in TEST_CELLS:
			if args.library == 'patski':
				dataFile = '../tools/newPickleData2/{0}_{1}MasterCellAlleleHapData.pickle'.format(cell, args.library)
			elif args.library == 'bj':
				dataFile = '../tools/bjPickleData/{0}_{1}MasterCellAlleleHapData.pickle'.format(cell, args.library)
			with open(dataFile, 'rb') as p1:
				print("Loading data from {0}".format(dataFile))
				masterCellDataDict = pickle.load(p1)
				vCounts = masterCellDataDict[args.cellName][0]
				vSites = masterCellDataDict[args.cellName][2]
				cellHaps.append(masterCellDataDict[cell][1])
				cellSites.append(masterCellDataDict[cell][3])

	# *** self.n_features = X.max() + 1 (https://github.com/hmmlearn/hmmlearn/issues/423)
	hapMap = {(1, 0): 0, (0, 1): 1, (1, 1): 2}
	
	cellPhasedList = [list() for c in TEST_CELLS]
	cellBinList = [list() for c in TEST_CELLS]
	print("c, len(chr), np.mean(chr), np.std(chr), ...")
	if not args.generate:
		for ci,cell in enumerate(TEST_CELLS):
			for c in chrList:
				phasedChrList = [hapMap[hi] for hi in cellHaps[ci][c]]
				if args.gaussian:
					cellPhasedList[ci].append([np.mean(phasedChrList[i:i+40]) for i in range(1, len(phasedChrList), 
														 BIN_FACTOR*cellCovDict[args.cellName])])
					cellBinList[ci].append([cellSites[ci][c][i] for i in range(1, len(cellSites[ci][c]), 
												   BIN_FACTOR*cellCovDict[args.cellName])])
					print(c, len(cellPhasedList[ci][-1]), np.mean(cellPhasedList[ci][-1]), np.std(cellPhasedList[ci][-1]), 
						cellPhasedList[ci][-1][:10])
				else:
					cellBinList[ci].append([cellSites[ci][c][i] for i in range(1, len(cellSites[ci][c]))])
					cellPhasedList[ci].append(phasedChrList)
					print(c, len(phasedChrList), "0:", phasedChrList.count(0), "1:", phasedChrList.count(1), "2:", phasedChrList.count(2))

	if args.gaussian:
		#hetState = None
		states, hetState = runGHmm(cellPhasedList, cellBinList, chrList, args)
		zTruth = None
	elif args.generate:
		simHmm(chrList, chrLenDict, COV)
	else:
		states, hetState = runHmm(cellPhasedList, cellSites, chrList, args)
		zTruth = None
	
	plotPatInline(vCounts, cellHaps[0], vSites, cellSites[0], chrLenDict, args.cellName, centDict, BIN_SIZE, BIN_SCALE, VAR_PER_BIN, 
		      chrList, args.library, states, cellBinList[0], hetState, zTruth)

if __name__ == "__main__":
        main()
