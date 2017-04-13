#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

	public:
		Alphabet wordAlpha; // should be initialized outside
		LookupTable words; // should be initialized outside

		vector<Alphabet> typeAlphas; // should be initialized outside
		vector<LookupTable> types;  // should be initialized outside

		RNNParams left_rnn_project;
		RNNParams right_rnn_project;
		BiParams Bi_hidden;
		UniParams olayer_linear; // output


	public:
		Alphabet labelAlpha; // should be initialized outside
		SoftMaxLoss loss;


	public:
		bool initial(HyperParams& opts, AlignedMemoryPool* mem){

			// some model parameters should be initialized outside
			if (words.nVSize <= 0 || labelAlpha.size() <= 0){
				return false;
			}
			opts.wordDim = words.nDim;
			opts.unitsize = opts.wordDim;
			opts.typeDims.clear();
			opts.labelSize = labelAlpha.size();

			left_rnn_project.initial(opts.hiddensize, opts.unitsize, mem);
			right_rnn_project.initial(opts.hiddensize, opts.unitsize, mem);
			Bi_hidden.initial(opts.hiddensize, opts.hiddensize, opts.hiddensize, true, mem);
			olayer_linear.initial(opts.labelSize, opts.hiddensize, false, mem);

			//		loss.initial(opts.labelSize);

			return true;
		}


		void exportModelParams(ModelUpdate& ada){
			words.exportAdaParams(ada);
			for (int idx = 0; idx < types.size(); idx++){
				types[idx].exportAdaParams(ada);
			}
			left_rnn_project.exportAdaParams(ada);
			right_rnn_project.exportAdaParams(ada);
			Bi_hidden.exportAdaParams(ada);
			olayer_linear.exportAdaParams(ada);
		}


		void exportCheckGradParams(CheckGrad& checkgrad){
			checkgrad.add(&(words.E), "_words.E");
			for (int idx = 0; idx < types.size(); idx++){
				stringstream ss;
				ss << "types[" << idx << "].E";
				checkgrad.add(&(types[idx].E), ss.str());
			}
			checkgrad.add(&(Bi_hidden.W1), "Bi_hidden.W1");
			checkgrad.add(&(Bi_hidden.W2), "Bi_hidden.W2");
			checkgrad.add(&(Bi_hidden.b), "Bi_hidden.b");

			checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		}

		// will add it later
		void saveModel(){

		}

		void loadModel(const string& inFile){

		}

};

#endif /* SRC_ModelParams_H_ */
