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
 * TODO docs
 */
public interface IOutputLayer extends ILayer {

  /**
   * Extends the forwardPass from ILayer to also calculate error and totalError in this pass.
   * So there is no need to use computeCostGradient or computeCost after this. Just use get...()
   * @see fko.nnplayground.API.ILayer#forwardPass(INDArray)
   */
  INDArray forwardPass(INDArray activationPreviousLayer);

  /**
   *
   * @param nExamples
   * @param training whether we compute this during training or outside of training
   * @return the array of errors for each example
   */
  INDArray computeCostGradient(INDArray labels, final int nExamples, boolean training);

  /**
   *
   * @param columns
   * @param training whether we compute this during training or outside of training
   * @return the array of errors for all examples (total loss)
   */
  double computeCost(INDArray labels, final int columns, boolean training);

  /**
   * Uses the internal error calculated based on the labels
   * @see ILayer#backwardPass(INDArray)
   * @return
   */
  INDArray backwardPass();

  INDArray getLabels();

}
