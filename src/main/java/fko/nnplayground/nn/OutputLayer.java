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
 */
public class OutputLayer extends Layer implements IOutputLayer {

  private static final Logger LOG = LoggerFactory.getLogger(OutputLayer.class);

  private INDArray labels;
  private INDArray error;
  private double totalError;

  public OutputLayer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit, final Activation.Activations activationFunction, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, seed);
  }

  @Override
  public INDArray forwardPass(final INDArray outputLastLayer) {
    super.forwardPass(outputLastLayer);
    return getActivation();
  }

  @Override
  public INDArray backwardPass() {
    super.backwardPass(error);
    return getPreviousLayerError();
  }

  @Override
  public INDArray computeError(final INDArray labels, final boolean training) {
    // TODO: this could be cached
    this.labels = labels;
    error = getActivation().sub(labels);
    totalError = Transforms.abs(error).meanNumber().doubleValue();
    return error;
  }

  @Override
  public double computeTotalError(final INDArray labels, final boolean training) {
    // TODO: this could be cached
    this.labels = labels;
    error = getActivation().sub(labels);
    totalError = Transforms.abs(error).meanNumber().doubleValue();
    return totalError;
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
