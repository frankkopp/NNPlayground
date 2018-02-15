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
import org.nd4j.linalg.ops.transforms.Transforms;
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

  public OutputLayer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
                     final Activation.Activations activationFunction, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, seed);
  }

  public OutputLayer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit,
                     final Activation.Activations activationFunction, double regStrength, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, regStrength, seed);
  }

  @Override
  public INDArray forwardPass(final INDArray activationPreviousLayer) {
    super.forwardPass(activationPreviousLayer);
    // TODO: if SOFTMAX compute class props
    return getActivation();
  }

  @Override
  public INDArray backwardPass() {
    super.backwardPass(cDelta);
    return getPreviousLayerDelta();
  }

  @Override
  public INDArray computeCostGradient(final INDArray labels, final int nExamples, final boolean training) {
    // TODO: this could be cached
    // TODO: implement different loss functions C

    this.labels = labels;

    // this is the gradient/derivative of the quadratic cost function!
    // (aL−y)- the derivative of cost also has a 1/n - this is done in the weights update to also cover the Bias
    // TODO: do we need to add regularization here?
    cDelta = getActivation().sub(this.labels);

    return cDelta;
  }

  @Override
  public double computeCost(final INDArray labels, final int nExamples, final boolean training) {
    // TODO: this could be cached
    // TODO: add more cost functions

    // quadratic cost function
    // (1/2n) * ∑ ||y(x)-aL(x)||^2
    final double unRegCost =
            Transforms.pow(labels.sub(activation), 2).sumNumber().doubleValue() / (2 * nExamples);
    final double regularization =
            0.5 * (regLamba / nExamples) * (weightsMatrix.mul(weightsMatrix).sumNumber().doubleValue());
    this.cost = unRegCost + regularization;
    return cost;
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
