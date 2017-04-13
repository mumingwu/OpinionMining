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


		RNNBuilder left_rnn1;
		RNNBuilder right_rnn1;
		RNNBuilder left_rnn2;
		RNNBuilder right_rnn2;
		RNNBuilder left_rnn3;
		RNNBuilder right_rnn3;

		vector<BiNode> word_Bi_hidden1;
		vector<BiNode> word_Bi_hidden2;
		vector<BiNode> word_Bi_hidden3;

		vector<LinearNode> output;
		unordered_map<string, int> *words;

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
			left_rnn1.resize(sent_length);
			right_rnn1.resize(sent_length);
			left_rnn2.resize(sent_length);
			right_rnn2.resize(sent_length);
			left_rnn3.resize(sent_length);
			right_rnn3.resize(sent_length);
			word_Bi_hidden1.resize(sent_length);
			word_Bi_hidden2.resize(sent_length);
			word_Bi_hidden3.resize(sent_length);
			output.resize(sent_length);

		}

		inline void clear(){
			Graph::clear();
			word_inputs.clear();
			word_Bi_hidden1.clear();
			word_Bi_hidden2.clear();
			word_Bi_hidden3.clear();
			output.clear();
		}

	public:
		inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem){
			int maxsize = word_inputs.size();
			for (int idx = 0; idx < maxsize; idx++) {
				word_inputs[idx].setParam(&model.words);
				word_Bi_hidden1[idx].setParam(&model.Bi_hidden1);
				word_Bi_hidden2[idx].setParam(&model.Bi_hidden2);
				word_Bi_hidden3[idx].setParam(&model.Bi_hidden3);
				output[idx].setParam(&model.olayer_linear);
			}
			left_rnn1.init(&model.left_rnn_project1, opts.dropOut, true, mem);
			right_rnn1.init(&model.right_rnn_project1, opts.dropOut, false, mem);
			left_rnn2.init(&model.left_rnn_project2, opts.dropOut, true, mem);
			right_rnn2.init(&model.right_rnn_project2, opts. dropOut, false, mem);
			left_rnn3.init(&model.left_rnn_project3, opts.dropOut, true, mem);
			right_rnn3.init(&model.right_rnn_project3, opts.dropOut, false, mem);

			for (int idx = 0; idx < maxsize; idx++){
				word_inputs[idx].init(opts.wordDim, opts.dropOut, mem);
				word_Bi_hidden1[idx].init(opts.segHiddenSize, opts.dropOut, mem);
				word_Bi_hidden2[idx].init(opts.hiddenSize, opts.dropOut, mem);
				word_Bi_hidden3[idx].init(opts.charhiddenSize, opts.dropOut, mem);
				output[idx].init(opts.labelSize, opts.dropOut, mem);
			}
		}

	public:
		inline string extract_words(const string& temp_words){
			double Pro = 0.5;
			unordered_map<string, int>::iterator it = words->find(temp_words);
			if(it != words-> end() && it-> second == 1){
				double x = rand() / double(RAND_MAX);
				if(x > Pro)
					return unknownkey;
				else
					return temp_words;
			}
			return temp_words;
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
			if(bTrain == true){
				for (int idx = 0; idx < seq_size; idx++) {
					const Feature& feature = features[idx];
					//input
					word_inputs[idx].forward(this, extract_words(feature.words[0]));
				}
			}
			else{
				for(int idx = 0; idx < seq_size; idx++){
					const Feature& feature = features[idx];
					word_inputs[idx].forward(this, feature.words[0]);
				}
			}
			left_rnn1.forward(this, getPNodes(word_inputs, seq_size));
			right_rnn1.forward(this, getPNodes(word_inputs, seq_size));


			for (int idx = 0; idx < seq_size; idx++) {
				word_Bi_hidden1[idx].forward(this, &(left_rnn1._output[idx]), &(right_rnn1._output[idx]));
			}

			left_rnn2.forward(this, getPNodes(word_Bi_hidden1, seq_size));
			right_rnn2.forward(this, getPNodes(word_Bi_hidden1, seq_size));

			for(int idx = 0; idx < seq_size; idx++){
				word_Bi_hidden2[idx].forward(this, &(left_rnn2._output[idx]), &(right_rnn2._output[idx]));
			}

			left_rnn3.forward(this, getPNodes(word_Bi_hidden2, seq_size));
			right_rnn3.forward(this, getPNodes(word_Bi_hidden2, seq_size));

			for(int idx = 0; idx < seq_size; idx++){
				word_Bi_hidden3[idx].forward(this, &(left_rnn3._output[idx]), &(right_rnn3._output[idx]));
				output[idx].forward(this, &(word_Bi_hidden3[idx]));
				
			}



		}

};

#endif /* SRC_ComputionGraph_H_ */
