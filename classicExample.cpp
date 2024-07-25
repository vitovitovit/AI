#include "ANNv2.hh"


int main(void) {
//NN uci zbrajati brojeva

	srand(time(0));

	TrainData data(2, 1);


        //podaci zbrajanja brojeva
	for (float i = 0.1f; i < 0.5f; i += 0.1f)
		for (float j = 0.1f; j < 0.5f; j += 0.1f)
			data.input({ i, j }).output({ i + j });



	NN net({ 2, 4, 4, 1 });   // net layers: 2 artificial neurons in 1.layer, 4 in second, 4 in third, 1 output

	net.setTrainingData(data);

	net.train(2, 10000);  // learning rate=2, epochs=10000


	net.print();
	net.eval(data);

	return 0;
}
