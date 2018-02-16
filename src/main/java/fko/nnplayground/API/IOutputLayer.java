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

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * IOutputLayer interface
 * TODO Javadoc
 */
public interface IOutputLayer extends ILayer {

  /**
   * The backward pass propagates the layer's error back using the gradient to the previous layer.
   * It calculates the error for this layer which can be queried by <code>getError()</code>.
   * As this is an OutputLayer we calculated the input ourselves and we so not need a parameter.
   *
   * @return the delta of the previous layer
   * @see ILayer#backwardPass(INDArray)
   */
  INDArray backwardPass();

  /**
   * @param nExamples The number of examples used in this back propagation run (batch)
   * @return the array of gradients for each example
   */
  INDArray computeCostGradient(INDArray labels, final int nExamples);

  /**
   * @param nExamples The number of examples used in this back propagation run (batch)
   * @return the array of loss for all examples (total loss)
   */
  double computeCost(INDArray labels, final int nExamples);

  INDArray getLabels();

}
