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

package fko.nnplayground.nn;

import fko.nnplayground.API.ILayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Layer
 * TODO: Javadoc
 */
public class Layer implements ILayer {

  private static final Logger LOG = LoggerFactory.getLogger(Layer.class);

  private final int inputSize;
  private final int outputSize;
  private final int seed;

  private Activation.Activations activationFunction;

  private final WeightInitializer.WeightInit weightInit;

  protected INDArray weightsMatrix;
  protected INDArray biasMatrix;

  protected double regLamba = 1e-3d; // default

  protected INDArray z_output; // before the non linearity function
  protected INDArray activation; // after the non linearity
  protected INDArray error; // gradient for layer
  protected INDArray previousLayerDelta; // error on the previous layer

  public Layer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
               final Activation.Activations activationFunction, final int seed) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weightInit = weightInit;
    this.activationFunction = activationFunction;
    this.seed = seed;

    weightsMatrix = WeightInitializer
            .initWeights(this.weightInit, this.outputSize, this.inputSize, this.seed);
    biasMatrix = WeightInitializer
            .initWeights(WeightInitializer.WeightInit.ZERO, this.outputSize, 1, this.seed);

    LOG.info(
            "Created layer of type {}. InputSize: {} OutputSize: {} " +
            "WeightInit: {} Activation: {} L2RegularizationStrength: {} ",
            getClass().getSimpleName(), inputSize, outputSize, weightInit, activationFunction, regLamba);
  }

  public Layer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
               final Activation.Activations activationFunction, final double regStrength, final int seed) {
    this(inputSize, outputSize, weightInit, activationFunction, seed);
    this.regLamba = regStrength;
  }

  /**
   * http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
   *
   * @param activationPreviousLayer the input for the layer
   * @return activation of this layer
   *
   * @see ILayer#forwardPass(INDArray)
   */
  @Override
  public INDArray forwardPass(final INDArray activationPreviousLayer) {
    // z = Wx + b
    z_output = weightsMatrix.mmul(activationPreviousLayer).addColumnVector(biasMatrix);
    // a = nonLin(z)
    activation = Activation.applyActivation(activationFunction, z_output);
    return getActivation();
  }

  @Override
  public INDArray backwardPass(INDArray delta) {
    // σ′(zl)
    final INDArray derivative = Activation.applyDerivative(activationFunction, activation);
    // δl=((wl+1)T*δl+1) ⊙ σ′(zl)
    error = delta.mul(derivative);
    // δl-l=((wl)T*δl)
    previousLayerDelta = weightsMatrix.transpose().mmul(error);
    return getPreviousLayerDelta();
  }

  @Override
  public void updateWeights(final INDArray activationPreviousLayer, final int nExamples, final double learningRate) {
    // Vanilla SGD update with L2 regularization for now
    // TODO: more updater

    // full change of weights based on gradient and layer_1 z_output
    // regularization - learningRate*delta
    // (1−ηλ/n)*w     − η * ∂C0/∂w
    final INDArray W_fullUpdate = error.mmul(activationPreviousLayer.transpose()).div(nExamples);
    // multiplied with learning rate to adjust step size
    final INDArray W_ratedUpdate = W_fullUpdate.mul(learningRate);
    // update W2 (incl. regularization)
    weightsMatrix.muli(1 - ((learningRate * regLamba) / nExamples)).subi(W_ratedUpdate);

    final INDArray b_ratedUpdated = error.sum(1).mul(learningRate / nExamples);
    biasMatrix.subi(b_ratedUpdated);
  }

  @Override
  public int getSeed() {
    return this.seed;
  }

  @Override
  public int getInputSize() {
    return inputSize;
  }

  @Override
  public int getOutputSize() {
    return outputSize;
  }

  @Override
  public void setActivationFunction(final Activation.Activations function) {
    this.activationFunction = function;
  }

  @Override
  public Activation.Activations getActivationFunction() {
    return activationFunction;
  }

  @Override
  public INDArray getZ_output() {
    return z_output.dup();
  }

  @Override
  public INDArray getActivation() {
    return activation.dup();
  }

  @Override
  public INDArray getWeightsMatrix() {
    return weightsMatrix.dup();
  }

  @Override
  public INDArray getBiasMatrix() {
    return biasMatrix.dup();
  }

  @Override
  public void setBiasMatrix(final INDArray newBiasMatrix) {
    biasMatrix = newBiasMatrix;
  }

  @Override
  public void setWeightsMatrix(final INDArray newWeightMatrix) {
    weightsMatrix = newWeightMatrix;
  }

  @Override
  public INDArray getError() {
    return error.dup();
  }

  @Override
  public INDArray getPreviousLayerDelta() {
    return previousLayerDelta.dup();
  }

  @Override
  public void setRegLamba(final double regLamba) {
    this.regLamba = regLamba;
  }

  @Override
  public double getRegLamba() {
    return regLamba;
  }

  @Override
  public WeightInitializer.WeightInit getWeightInit() {
    return weightInit;
  }

  @Override
  public String toString() {
    return "Layer{" +
           "inputSize=" + inputSize +
           ", outputSize=" + outputSize +
           ", seed=" + seed +
           ", activationFunction=" + activationFunction +
           ", weightInit=" + weightInit +
           '}';
  }
}
