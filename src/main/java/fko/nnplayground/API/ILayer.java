/*
 * MIT License
 *
 * Copyright (c) 2018 Frank Kopp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package fko.nnplayground.API;

import fko.nnplayground.nn.Activation;
import fko.nnplayground.nn.WeightInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * ILayer interface
 * TODO Javadoc
 */
public interface ILayer {

  /**
   * The forward pass uses the layer input data to calculate the activation (output) of the layer.
   * It computes the raw weights*input (Wx+b) value called z_output and the activation (usually called a)
   * which is computed using a non linear function like ReLU, Sigmoid, etc.
   * <p>
   * http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
   *
   * @param activationPreviousLayer the input for the layer
   * @return activation of this layer
   */
  INDArray forwardPass(INDArray activationPreviousLayer);

  /**
   * @return the z_output of the layer before the nonLin activation function
   */
  INDArray getZ_output();

  /**
   * The backward pass propagates the layer's error back using the gradient to the previous layer.
   * It calculates the error for this layer which can be queried by <code>getError()</code>.
   *
   * @param delta of this layer (back propagated from next layer)
   * @return the delta of the previous layer
   */
  INDArray backwardPass(INDArray delta);

  /**
   * After the backward pass through all layers is complete this method is called
   * on all layers to update the weights of this layer.
   *
   * @param activationPreviousLayer the input from the previous layer
   * @param nExamples the number of examples used in the backpropagation run (usually batch size)
   * @param learningRate the factor for update steps on the weights
   */
  void updateWeights(final INDArray activationPreviousLayer, final int nExamples, double learningRate);

  /*
   * Getters and Setters
   */

  WeightInitializer.WeightInit getWeightInit();

  INDArray getActivation();

  INDArray getWeightsMatrix();

  INDArray getBiasMatrix();

  void setBiasMatrix(INDArray read_biasMatrix);

  void setWeightsMatrix(INDArray read_weightMatrix);

  void setActivationFunction(Activation.Activations function);

  Activation.Activations getActivationFunction();

  int getSeed();

  int getInputSize();

  int getOutputSize();

  INDArray getError();

  INDArray getPreviousLayerDelta();

  /**
   * Regularization strength
   *
   * @param l2Strength set regularization strength. Default 0.001. Set to 0 to turn off regularization.
   */
  void setL2Strength(double l2Strength);

  double getL2Strength();
}
