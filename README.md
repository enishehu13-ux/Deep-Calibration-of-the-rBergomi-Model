This project implements an Artificial Neural Network (ANN) to efficiently calibrate the rough Bergomi (rBergomi) volatility model.

Project Files & Structure:

MC sims.py: Generates a synthetic training dataset of options priced under the rBergomi model, utilizing the turbocharged and random grids methods (refer to the associated papers for mathematical details).

Train.py: Configures and trains the ANN, which is built with a deep learning architecture of 4 hidden layers and 64 neurons per layer.

DE calibration example.py: Provides a practical demonstration of using the trained ANN for model calibration. You can calibrate the model to any target volatility surface by simply swapping out the input .csv file.

rbergomi_ann.keras & rbergomi_scaler.pkl: These files store the pre-trained model artifacts required for inference. They contain the network's learned weights, biases, and the specific scaling transformations needed for the rBergomi parameters to function correctly with the ANN.
