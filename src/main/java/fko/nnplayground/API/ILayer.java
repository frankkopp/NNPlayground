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
 * TODO docs
 */
public interface ILayer {

  /**
   * Computes the layer's forward pass
   * @param outputLastLayer the input for the layer
   * @return the activation of the layer (after activation function)
   */
  INDArray forwardPass(INDArray outputLastLayer);

  /**
   * @return the z_output of the layer before the nonLin activation function
   */
  INDArray getZ_output();

  /**
   * @param error of this layer (back propagated from next layer)
   * @return the error of the previous layer
   */
  INDArray backwardPass(INDArray error);

  /**
   * @param activationPreviousLayer
   * @param nExamples
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

  void setActivationFunction(Activation.Activations function);
  Activation.Activations getActivationFunction();

  int getSeed();

  int getInputSize();
  int getOutputSize();

  INDArray getError();

  INDArray getPreviousLayerError();

  /**
   * Regularization strength
   * @param regLamba set regularization strength. Default 0.001. Set to 0 to turn off regularization.
   */
  void setRegLamba(double regLamba);
  double getRegLamba();
}
