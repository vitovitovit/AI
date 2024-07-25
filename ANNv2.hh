


#ifndef LIBS

#include <assert.h>
#include <iostream>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <time.h>

#define LIBS

#endif // LIBS



template<typename T>
class Matrix {    // klasa za rad s matricama


private:

	std::vector<T> matrix;
	size_t rows;    // number of rows
	size_t cols;    // number of cols
	size_t count;    // number of elements


public:

	Matrix(size_t r, size_t c) : rows(r), cols(c) {

		matrix; //= new T[rows * cols];
		count = rows * cols;

		if constexpr (std::is_fundamental<T>::value) {
			for (size_t i = 0; i < count; ++i)
				//matrix[i] = 0;
				matrix.push_back(0);
		}
		else if constexpr (std::is_pointer<T>::value) {
			for (size_t i = 0; i < count; ++i)
				//matrix[i] = nullptr;
				matrix.push_back(nullptr);
		}
		else {
			for (size_t i = 0; i < count; ++i)
				//matrix[i] = T();
				matrix.push_back(T());
		}
	}


	static void matMul(Matrix<T>& res,  Matrix<T>& a, Matrix<T>& b) {    // matrix multiplication, res = a * b

		if (a.getCols() != b.getRows())
			throw std::invalid_argument("in Matrix<T>::matMul, a or b matrix is in wrong shape!");

		if (a.getRows() != res.getRows() || b.getCols() != res.getCols())
			throw std::invalid_argument("in Matrix<T>::matMul, result matrix is in wrong shape!");

		for (size_t i = 0; i < res.getRows(); ++i) {
			for (size_t j = 0; j < res.getCols(); ++j) {
				res.at(i, j) = 0; // Inicijalizacija na 0
				for (size_t k = 0; k < a.getCols(); ++k) {
					res.at(i, j) += a.at(i, k) * b.at(k, j);
				}
			}
		}
	}

	static void matSum(Matrix<T>& res, Matrix<T>& a, Matrix<T>& b) {    // matrix sumation, res = a + b

		if (a.getRows() != b.getRows() && b.getRows() != res.getRows())
			throw std::invalid_argument("in Matrix<T>::matSum(), num of rows is wrong!");

		else if (a.getCols() != b.getCols() && b.getCols() != res.getCols())
			throw std::invalid_argument("in Matrix<T>::matSum, num of cols is wrong!");

		for (size_t i = 0; i < res.getRows(); ++i) {
			for (size_t j = 0; j < res.getCols(); ++j) {
				res.at(i, j) = a.at(i, j) + b.at(i, j);
			}
		}
	}

	static void matSub(Matrix<T>& res, Matrix<T>& a, Matrix<T>& b) {

		if (a.getRows() != b.getRows() && b.getRows() != res.getRows())
			throw std::invalid_argument("in Matrix<T>::matSum(), num of rows is wrong!");

		else if (a.getCols() != b.getCols() && b.getCols() != res.getCols())
			throw std::invalid_argument("in Matrix<T>::matSum, num of cols is wrong!");

		for (size_t i = 0; i < res.getRows(); ++i) {
			for (size_t j = 0; j < res.getCols(); ++j) {
				res.at(i, j) = a.at(i, j) - b.at(i, j);
			}
		}

	}

	size_t getCount() {
		return count;
	}

	size_t size() {
		return count;
	}

	size_t getRows() {    // num of rows
		return rows;
	}

	size_t getCols() {    // num of cols
		return cols;
	}

	void setMatrix(std::initializer_list<T> initList) {   // example: .setMatrix({1, 0,  0, 1});  // depend on num of rows and cols that are setted

		if (initList.size() > count) {
			throw std::invalid_argument("in Matrix<T>::setMatrix(), Initializer list size does not match matrix size");
		}

		size_t i = 0;

		for (const auto& elem : initList) {
			matrix[i++] = elem;
		}

	}




	static Matrix<T> Vec2Mat(std::vector<T> vec) {

		Matrix<T> mat(vec.size(), 1);


		for (size_t i = 0; i < vec.size(); ++i) {
			mat.at(i) = vec.at(i);
		}

		return mat;


	}

	static std::vector<T> Mat2Vec(Matrix<T> mat) {

		std::vector<T> vec(mat.getCount());

		for (size_t i = 0; i < mat.getCount(); ++i) {
			vec.at(i) = mat.at(i);
		}


		return vec;


	}


	Matrix<T> getTranspose() {

		Matrix<T> transpose(this->getCols(), this->getRows());

		for (size_t i = 0; i < this->getRows(); ++i) {
			for (size_t j = 0; j < this->getCols(); ++j) {
				transpose.at(j, i) = this->at(i, j);
			}
		}



		return transpose;


	}

	void fill(float x) {
		for (size_t i = 0; i < this->getRows(); ++i) {

			this->matrix.at(i) = x;

		}
	}


	T get(size_t row, size_t col) {   // no this, better use at(i, j)

		if (row >= rows || col >= cols)
			throw std::invalid_argument("row or col > scope, in matrix.get()");

		return matrix[row * cols + col];
	}

	void set(size_t row, size_t col, T value) {    // no this, better use at(i, j)

		if (row >= rows || col >= cols)
			throw std::invalid_argument("row or col > scope, in matrix.set()");

		matrix[row * cols + col] = value;


	}

	static void scalarMul(Matrix<T>& A, float x) {

		for (size_t i = 0; i < A.getRows(); ++i) {
			for (size_t j = 0; j < A.getCols(); ++j) {
				A.at(i, j) *= x;
			}
		}

	}

	static Matrix<float> naiveMul( Matrix<T>& A, Matrix<T>& B) {


	/*	if (A.getCols() != B.getCols())
			throw std::invalid_argument("in Matrix<float>::naiveMul wrong nums of cols in A or B");
		if (A.getRows() != B.getRows)
			throw std::invalid_argument("in Matix<float>::naiveMul wrong nums of rows in A or B");*/

		Matrix<float> res(A.getRows(), A.getCols());

		for (size_t i = 0; i < res.getRows(); ++i) {
			for (size_t j = 0; j < res.getCols(); ++j) {
				res.at(i, j) = A.at(i, j) * B.at(i, j);
			}
		}

		return res;

	}

	friend Matrix<T> operator* (Matrix<T> A, Matrix<T> B) {

		Matrix<T> res(A.getRows(), B.getCols());

		Matrix<T>::matMul(res, A, B);

		return res;


	}

	// ako ne radi vratiti & na sve parametre u overloadima Matrix&

	friend Matrix<T> operator* (float scalar, Matrix<T> A) {

		Matrix<T> res(A.getRows(), A.getCols());

		 Matrix<T>::scalarMul(A, scalar);

		 res = A;

		return res;
	}

	friend Matrix<T> operator* (Matrix<T> A, float scalar) {
	
		Matrix<T> res(A.getRows(), A.getCols());

		Matrix<T>::scalarMul(A, scalar);

		return res;
	}

	friend Matrix<T> operator+ (Matrix<T> A, Matrix<T> B) {

		Matrix<T> res(A.getRows(), B.getCols());

		Matrix<T>::matSum(res, A, B);

		return res;
	}

	friend Matrix<T> operator- (Matrix<T> A, Matrix<T> B) {

		Matrix<T> res(A.getRows(), B.getCols());

		Matrix<T>::matSub(res, A, B);

		return res;
	}

	T& at(size_t row, size_t col) {   // same thing as matrix[i][j]  // example of usage: .at(i, j)++;

		if (row >= rows || col >= cols)
			throw std::invalid_argument("in matrix.at(), row or col > scope");

		return matrix[row * cols + col];
	}

	T& at(size_t index) {
		if (index >= this->count)
			throw std::invalid_argument("at(index) wrong argument, index too big");
		else if (index < 0)
			throw std::invalid_argument("at(index) wrong argument, index can't be negative");
		else
			return matrix[index];
	}

	void print() {    // print whole matrix

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				std::cout << matrix[i * cols + j] << " ";
			}

			std::cout << std::endl;
		}
	}

	std::vector<T> vectorize() {

		if (this->getCols() != 1)
			throw std::invalid_argument("from Matrix<T>::vectorize() -- matrix is not vector, can't be casted to std::vector<T>");


		else
			return this->matrix;
	}

/*
~Matrix() {
		delete[] matrix;
	}*/

};


class TrainData {    // klasa za rad sa podacima za trening AI

private:
	size_t input_size;    // size of input vector
	size_t output_size;    // size of output vector
	size_t samples_count;    // number of training data samples

	std::vector<std::vector<float>> inputs;
	std::vector<std::vector<float>> outputs;



public:

	TrainData(size_t in, size_t out) : input_size(in), output_size(out) {

	}

	TrainData() {

	}

	std::vector<float>& getInput(size_t index) {
		return inputs[index];
	}

	std::vector<float>& getOutput(size_t index) {
		return outputs[index];
	}

	std::vector<std::vector<float>>& getAllInputs() {
		return inputs;
	}

	std::vector<std::vector<float>>& getAllOutputs() {
		return outputs;
	}

	bool isEmpty() {    // napraviti

		return true;
	}



	void inputSize(size_t in){

		input_size = in;

	}

	void outputSize(size_t out) {

		output_size = out;

	}

	TrainData& input(std::initializer_list<float> in) {
		if (in.size() != input_size) {
			throw std::invalid_argument("in TrainDescription::input({...}), {...}.size() is different than setted input size inputSize(size_t in)!");
		}

		inputs.push_back(in);

		return *this;
	}

	TrainData& output(std::initializer_list<float> out) {

		if (out.size() != output_size) {
			throw std::invalid_argument("in TrainDescription::input({}).output({...}), {...}.size() is different than setted output size outputSize(size_t out)!");
		}

		outputs.push_back(out);

		samples_count = outputs.size();

		return *this;

	}


	TrainData& input(std::vector<float> in) {
		if (in.size() != input_size) {
			throw std::invalid_argument("in TrainDescription::input({...}), {...}.size() is different than setted input size inputSize(size_t in)!");
		}

		inputs.push_back(in);

		return *this;
	}

	TrainData& output(std::vector<float> out) {

		if (out.size() != output_size) {
			throw std::invalid_argument("in TrainDescription::input({}).output({...}), {...}.size() is different than setted output size outputSize(size_t out)!");
		}

		outputs.push_back(out);

		samples_count = outputs.size();

		return *this;

	}

	bool confirm() {
		if (inputs.size() != outputs.size())
			return false;
			//throw std::invalid_argument("confirm() -- can't confirm because not every input has output");
		else {
			samples_count = inputs.size();
			return true;
		}

	}

	size_t size() {
		return samples_count;
	}

	void print() {

		for (size_t i = 0; i < size(); ++i) {
			std::cout << "{ ";

			for (auto input : inputs[i])
				std::cout << input << ", ";

			std::cout << " } --> { ";

			for (auto output : outputs[i])
				std::cout << output << ", ";

			std::cout << " }\n";

		}

	}




};


class NN {   // Artificial NN model


private:
	// forward fields
	std::vector<size_t> layers;
	std::vector<Matrix<float>> weights;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> activations;


	// training fields
	TrainData trainData;



public:



	static float randomFloat(float min, float max) {    // random number generator between min and max, can be used to init NN model

		float random = (float)rand() / (float)RAND_MAX;
		return min + random * (max - min);
	}

	NN(std::initializer_list<size_t> initList) {    // generate W, B matrices on description, for example: {2, 3, 3, 1} generate ANN of 2 input ANs in I.layer, 2 hidden layers, 1 output

		for (auto layer : initList) {
			layers.push_back(layer);
		}



		activations.push_back(Matrix<float>(this->layers[0], 1));   //  a0 = input activation matrix


		for (size_t i = 1; i < this->layers.size(); ++i) {
			size_t rows = this->layers[i ];
			size_t cols = this->layers[i - 1];



			Matrix<float> w = Matrix<float>(rows, cols);
			Matrix<float> b = Matrix<float>(rows, 1);
			Matrix<float> a = Matrix<float>(rows, 1);

			weights.push_back(w);
			biases.push_back(b);
			activations.push_back(a);

			// fill weights and biases with random values
			for (size_t j = 0; j < rows; ++j) {
				for (size_t k = 0; k < cols; ++k) {
					weights[i - 1].at(j, k) = NN::randomFloat(-10, 10);
				}
				biases[i - 1].at(j, 0) = NN::randomFloat(-10, 10);

			}
		}



	}

	void input(std::initializer_list<float> initList) {

		if (initList.size() != activations[0].getRows())
			throw std::invalid_argument("incorect input vector! error in NN::input({}) method! not same number x of elements in input vector as in NN definition ({x, ...})");


		Matrix<float>& M = activations[0];

		size_t i = 0;

		for (auto elem : initList) {
			M.at(i, 0) = elem;
			++i;
		}

	}

	void input(std::vector<float> initList) {

		if (initList.size() != activations[0].getRows())
			throw std::invalid_argument("incorect input matrix! error in NN::input({}) method!");


		Matrix<float>& M = activations[0];

		size_t i = 0;

		for (auto elem : initList) {
			M.at(i, 0) = elem;
			++i;
		}

	}

	void activate(Matrix<float>& Z) {

		for (size_t i = 0; i < Z.getRows(); ++i) {
			for (size_t j = 0; j < Z.getCols(); ++j) {
				Z.at(i, j) = sigmoid(Z.at(i, j));
			}
		}


	}








	void forward() {


		for (size_t i = 0; i < activations.size() - 1; ++i) {

			activations[i + 1] = weights[i] * activations[i];
			activations[i + 1] = activations[i + 1] + biases[i];

			activate(activations[i + 1]);


		}



	}

	Matrix<float> output() {


		return activations[activations.size() - 1];


	}


	void print() {

		//std::cout << "________\n";


		std::cout << "\n_________\nINPUT LAYER 0: \n";

		activations[0].print();


		for (size_t i = 0; i < weights.size(); ++i) {


			if (i == weights.size() - 1)
				std::cout << "\n_________\nOUTPUT LAYER: \n";
			/*else if (i == 0)
				std::cout << "\n_________\nHIDDEN: \n";*/
			else
				std::cout << "\n_________\n" << i + 1 << ". HIDDEN LAYER \n";

			std::cout << "\n\n";

			std::cout << "input A " << i << ": {\n";

			activations[i].print();
			std::cout << "}\n\n";

			std::cout << "w " << i  + 1<< ": {\n";
			weights[i].print();
			std::cout << "}\n\n";

			std::cout << "b " << i + 1 << ": {\n";
			biases[i].print();
			std::cout << "}\n\n";

			std::cout << "output of " << i + 1 << ". LAYER" << "\nA " << i + 1 << ": {\n";
			activations[i + 1].print();
			std::cout << "}\n\n";





		}

		std::cout << "________\n";

	}

	// TRAINING MODEL

	static float sigmoid(float x) {   // activation

		return (float)1.0f / (1.0f + std::exp(-x));

	}

	static float sigmoidDerivative(float sigma) {
		return sigma * (1.0f - sigma);
	}

	static void sigmaPrime(Matrix<float>& Z) {

		for (size_t i = 0; i < Z.getCount(); ++i) {
			Z.at(i) = NN::sigmoidDerivative(Z.at(i));
		}

	}

	void setTrainingData(TrainData data) {



		if (!data.confirm())
			throw std::invalid_argument("TrainData.confirm() -- from setTrainingData(TrainData data), not valid Data format, not same number of inputs and outputs!");
		else
			trainData = data;



	}

	Matrix<float> diagonalPrime(Matrix<float>& A) {

		Matrix<float> Zprime(A.getRows(), A.getRows());

		for (size_t i = 0; i < Zprime.getRows(); ++i) {
			for (size_t j = 0; j < Zprime.getCols(); ++j) {
				if (i != j)
					Zprime.at(i, j) = 0;
				else
					Zprime.at(i, j) = NN::sigmoidDerivative(A.at(i));
			}
		}


		return Zprime;

	}


	void eval(TrainData testData) {

	   std::vector<std::vector<float>> outs =	testData.getAllOutputs();
	   std::vector<std::vector<float>> inps = testData.getAllInputs();

	   for (size_t i = 0; i < outs.size(); ++i) {
		   this->input(inps[i]);
		   this->forward();
		   Matrix<float> netOut = this->output();


		 

		  Matrix<float> inp = Matrix<float>::Vec2Mat(inps[i]);
		  inp.print();

		   std::cout << " -->\n";

		// Matrix<float> targetOut = Matrix<float>::Vec2Mat(outs[i]);

		
		  netOut.print();

		  std::cout << "\n\n";
		  /*
		  std::cout << "\nerror: \n";

		  (netOut - targetOut).print();

		  std::cout << "\n_____";
		  */
	   }

	}


	void train(float learningRate, int epochs) {    // training by backpropagation algorithm
		if (!trainData.isEmpty())
			throw std::logic_error("Training data is not set.");

		for (int epoch = 0; epoch < epochs; ++epoch) {
			float totalCost = 0.0f;

			// for all training samples from data set
			for (size_t n = 0; n < trainData.size(); ++n) {
				std::vector<float> input = trainData.getInput(n);
				std::vector<float> targetOutput = trainData.getOutput(n);

				// FEED FORWARD
				this->input(input);
				this->forward();
				Matrix<float> output = this->output();

				//  LAST DELTA CALCULATION
				Matrix<float> matTargetOutput = Matrix<float>::Vec2Mat(targetOutput);
				Matrix<float> outputError = output - matTargetOutput;  //Matrix<float>::Vec2Mat(targetOutput);
				outputError = outputError.getTranspose();   // 1xn matrix
				Matrix<float> diagPrime = NN::diagonalPrime(activations.back());
				Matrix<float> delta = outputError * diagPrime; // diagonalPrime(activations.back());

				Matrix<float> deltaTrans = delta.getTranspose();
				Matrix<float> aTransp = activations[activations.size() - 2].getTranspose();

				Matrix<float> firstDW = (deltaTrans * aTransp);

				firstDW = learningRate * firstDW;

		


				// UPDATING LAST WEIGHT
				weights.back() = weights.back() - firstDW;		


				// BACKPROPAGATION (CALCULATING ALL DELTAS AND UPDATING WEIGHTS)
				
				for (int layer = activations.size() - 2; layer >= 1; --layer) {  // od predzadnjeg layera, do layera sa a1, to je zadnja delta (dA1/dZ1)

				

					Matrix<float> prevActivation = activations[layer];
					Matrix<float> prevActivationDiff = NN::diagonalPrime(prevActivation); 
					Matrix<float> weight = weights[layer];
					
				
					Matrix<float> prevDelta = delta * weight;
					prevDelta = prevDelta * prevActivationDiff;


					// Update weights and biases by formula:  delta.transpose * activation.transpose (dC/dZ * dZ/dW)

					Matrix<float> prevDeltaT = prevDelta.getTranspose();
					Matrix<float> aTranspose = activations[layer - 1].getTranspose();


						

					Matrix<float> deltaWeight = prevDeltaT * aTranspose;	
					Matrix<float> deltaBias = prevDeltaT;


					Matrix<float> dW = (learningRate * deltaWeight);
					Matrix<float> dB = (learningRate * deltaBias);
		

					weights[layer - 1] = weights[layer - 1] - dW;
					biases[layer - 1] = biases[layer - 1] - dB;

					delta = prevDelta;

				
				}

			
			}

			// Print average cost per epoch
			//totalCost /= trainData.size();
			std::cout << "Epoch " << epoch + 1 << ", Average Cost: " << this->getCost() << std::endl;
		}
	}


	

	// BEGIN COST


	float getCost() {
		std::vector<std::vector<float>> inputs = trainData.getAllInputs();
		std::vector<std::vector<float>> outputs = trainData.getAllOutputs();

		float totalCost = 0;

		for (size_t n = 0; n < outputs.size(); ++n) {
			std::vector<float> wantedOut = outputs[n];
			std::vector<float> in = inputs[n];

			this->input(in);
			this->forward();
			std::vector<float> netOut = Matrix<float>::Mat2Vec(this->output());

			if (wantedOut.size() != netOut.size())
				throw std::invalid_argument("Output vector size mismatch in getCost()!");

			float sampleCost = 0;
			for (size_t i = 0; i < wantedOut.size(); ++i) {
				sampleCost += (netOut[i] - wantedOut[i]) * (netOut[i] - wantedOut[i]);
			}

			sampleCost /= wantedOut.size();
			totalCost += sampleCost;
		}

		totalCost /= outputs.size();

		return totalCost;
	}


	// private
	float singlevariableCost(float activation, float wanted) {    // cost of 1 atribute of activation vector and wanted vector
		return powf(wanted - activation, 2.0f);
	}


	// private
	float multivariableCost(std::vector<float> wantedVec, std::vector<float> activationVec) {    // total cost of activation vector and wanted vector, based on single training sample


		if (activationVec.size() != wantedVec.size())
			throw std::invalid_argument("from multivariableCost(std::vector<float> activationVec, std::vector<float> wantedVec)\n vectors of output and wanted output are different in size\n"
				"try to see if TrainData Data(x, y); has same size of output (y) as last (output layer) of NN (NN net({2, 3, 4, ..., y })");


		float error = 0;

		size_t output_size = activationVec.size();

		for (size_t i = 0; i < output_size; ++i) {
			error += singlevariableCost(activationVec[i], wantedVec[i]);
		}


		error /= (float)output_size;

		return error;



	}

	float totalCost(TrainData data) {    // total cost based on all training sample from data

		std::vector<std::vector<float>> inputs = data.getAllInputs();
		std::vector<std::vector<float>> outputs = data.getAllOutputs();

		float totalError = 0.0f;

		for (size_t i = 0; i < data.size(); ++i) {

			input(inputs[i]);

			forward();

			totalError += multivariableCost(outputs[i], this->output().vectorize());

		}

		totalError /= (float)data.size();

		return totalError;

	}



	float totalCost() {    // total cost based on all training sample from data (but arg is given data)!




		TrainData data = this->trainData;



		std::vector<std::vector<float>> inputs = data.getAllInputs();
		std::vector<std::vector<float>> outputs = data.getAllOutputs();

		float totalError = 0.0f;

		for (size_t i = 0; i < data.size(); ++i) {

			input(inputs[i]);

			forward();



			totalError += multivariableCost(outputs[i], this->output().vectorize());

		}

		totalError /= (float)data.size();

		return totalError;

	}

	Matrix<float> totalCostDerivative() {

		// total cost = ( vecA - vecWA )^2  --> derivative --> cost` = 2 * (vecA - vecWA )

		trainData.getAllOutputs();


	}

	// END COST


};


