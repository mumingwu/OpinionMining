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

		RNNParams left_rnn_project1;
		RNNParams right_rnn_project1;
		RNNParams left_rnn_project2;
		RNNParams right_rnn_project2;
		RNNParams left_rnn_project3;
		RNNParams right_rnn_project3;

		BiParams Bi_hidden1;
		BiParams Bi_hidden2;
		BiParams Bi_hidden3;
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

			left_rnn_project1.initial(opts.segHiddenSize, opts.unitsize, mem);
			right_rnn_project1.initial(opts.segHiddenSize, opts.unitsize, mem);
			left_rnn_project2.initial(opts.hiddenSize, opts.segHiddenSize, mem);
			right_rnn_project2.initial(opts.hiddenSize, opts.segHiddenSize, mem);
			left_rnn_project3.initial(opts.rnnHiddenSize, opts.hiddenSize, mem);
			right_rnn_project3.initial(opts.rnnHiddenSize, opts.hiddenSize, mem);
			Bi_hidden1.initial(opts.segHiddenSize, opts.segHiddenSize, opts.segHiddenSize, true, mem);
			Bi_hidden2.initial(opts.hiddenSize, opts.hiddenSize, opts.hiddenSize, true, mem);
			Bi_hidden3.initial(opts.charhiddenSize, opts.rnnHiddenSize, opts.rnnHiddenSize, true, mem);
			olayer_linear.initial(opts.labelSize, opts.charhiddenSize, false, mem);

			//		loss.initial(opts.labelSize);

			return true;
		}


		void exportModelParams(ModelUpdate& ada){
			words.exportAdaParams(ada);
			for (int idx = 0; idx < types.size(); idx++){
				types[idx].exportAdaParams(ada);
			}
			left_rnn_project1.exportAdaParams(ada);
			right_rnn_project1.exportAdaParams(ada);
			left_rnn_project2.exportAdaParams(ada);
			right_rnn_project2.exportAdaParams(ada);
			left_rnn_project3.exportAdaParams(ada);
			right_rnn_project3.exportAdaParams(ada);
			Bi_hidden1.exportAdaParams(ada);
			Bi_hidden2.exportAdaParams(ada);
			Bi_hidden3.exportAdaParams(ada);
			olayer_linear.exportAdaParams(ada);
		}


		void exportCheckGradParams(CheckGrad& checkgrad){
			checkgrad.add(&(words.E), "_words.E");
			for (int idx = 0; idx < types.size(); idx++){
				stringstream ss;
				ss << "types[" << idx << "].E";
				checkgrad.add(&(types[idx].E), ss.str());
			}
			checkgrad.add(&(Bi_hidden1.W1), "Bi_hidden1.W1");
			checkgrad.add(&(Bi_hidden1.W2), "Bi_hidden1.W2");
			checkgrad.add(&(Bi_hidden1.b), "Bi_hidden1.b");
			
			checkgrad.add(&(Bi_hidden2.W1), "Bi_hidden2.W1");
			checkgrad.add(&(Bi_hidden2.W2), "Bi_hidden2.W2");
			checkgrad.add(&(Bi_hidden2.b), "Bi_hidden2.b");

			checkgrad.add(&(Bi_hidden3.W1), "Bi_hidden3.W1");
			checkgrad.add(&(Bi_hidden3.W2), "Bi_hidden3.W2");
			checkgrad.add(&(Bi_hidden3.b), "Bi_hidden3.b");

			checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		}

		// will add it later
		void saveModel(){

		}

		void loadModel(const string& inFile){

		}

};

#endif /* SRC_ModelParams_H_ */
