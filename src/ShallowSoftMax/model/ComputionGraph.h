#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	// vector<vector<LookupNode> > word_inputs;
	vector<LookupNode> word_inputs;


	RNNBuilder left_rnn;
	RNNBuilder right_rnn;

	vector<LinearNode> output;

	int type_num;


	// node pointers
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){
		type_num = typeNum;
		word_inputs.resize(sent_length);
		left_rnn.resize(sent_length);
		output.resize(sent_length);

	}

	inline void clear(){
		Graph::clear();
		word_inputs.clear();
		output.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem){
		int maxsize = word_inputs.size();
		for (int idx = 0; idx < maxsize; idx++) {
			word_inputs[idx].setParam(&model.words);
			output[idx].setParam(&model.olayer_linear);
		}
		left_rnn.init(&model.left_rnn_project, opts.dropOut, mem);

		for (int idx = 0; idx < maxsize; idx++){
			word_inputs[idx].init(opts.wordDim, opts.dropOut, mem);
			output[idx].init(opts.labelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx].forward(this, feature.words[0]);
		}
		left_rnn.forward(this, getPNodes(word_inputs, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			output[idx].forward(this, &(left_rnn._output[idx]));
		}
	}

};

#endif /* SRC_ComputionGraph_H_ */
