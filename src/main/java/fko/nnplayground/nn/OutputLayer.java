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

import fko.nnplayground.API.IOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OutputLayer
 * TODO: Javadoc
 */
public class OutputLayer extends Layer implements IOutputLayer {

  private static final Logger LOG = LoggerFactory.getLogger(OutputLayer.class);

  private INDArray labels;
  private double cost;
  private INDArray cDelta;

  private LossFunction.LossFunctions lossFunction = LossFunction.LossFunctions.MSE;;

  public OutputLayer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
                     final Activation.Activations activationFunction, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, seed);
  }

  public OutputLayer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
                     final Activation.Activations activationFunction, double regStrength, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, regStrength, seed);
  }

  /**
   * @see fko.nnplayground.API.ILayer#forwardPass(INDArray)
   */
  @Override
  public INDArray forwardPass(final INDArray activationPreviousLayer) {
    super.forwardPass(activationPreviousLayer);
    return getActivation();
  }

  @Override
  /**
   * @see fko.nnplayground.API.ILayer#backwardPass(INDArray)
   */
  public INDArray backwardPass() {
    super.backwardPass(cDelta);
    return getPreviousLayerDelta();
  }

  @Override
  public double computeCost(final INDArray labels, final int nExamples) {
    // TODO: this could be cached
    // compute loss
    final double unRegCost = LossFunction.computeLoss(lossFunction, labels, activation, nExamples);
    // compute regularization
    final double regularization =
            0.5 * (l2Strength / nExamples) * (weightsMatrix.mul(weightsMatrix).sumNumber().doubleValue());
    // compute total loss
    this.cost = unRegCost + regularization;
    return cost;
  }

  @Override
  public INDArray computeCostGradient(final INDArray labels, final int nExamples) {
    // TODO: this could be cached
    this.labels = labels;
    this.cDelta = LossFunction.computeLossGradient(lossFunction, labels, activation, nExamples);
    return cDelta;
  }

  @Override
  public INDArray getLabels() {
    return labels;
  }

  @Override
  public String toString() {
    return "OutputLayer{" +
           super.toString() + " }";
  }
}
