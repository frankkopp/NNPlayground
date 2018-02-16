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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * WeightInitializer
 */
public class WeightInitializer {

  private static final Logger LOG = LoggerFactory.getLogger(WeightInitializer.class);

  public static INDArray initWeights(final WeightInit weightInit, final int outputSize, final int inputSize, final int seed) {
    final INDArray weightsMatrix;
    switch (weightInit) {
      case XAVIER:
        // as per Dl4j
        weightsMatrix = Nd4j.randn(outputSize, inputSize, seed).mul(FastMath.sqrt(2.0 / (inputSize + outputSize)));
        break;
      case XAVIER_FAN_IN:
        weightsMatrix = Nd4j.randn(outputSize, inputSize, seed).div(FastMath.sqrt(inputSize));
        break;
      case XAVIER_LIKE:
        // XAVIER initialization:
        // as per CS231n
        weightsMatrix = Nd4j.randn(outputSize, inputSize, seed).div(FastMath.sqrt(inputSize / 2));
        break;
      case XAVIER_UNIFORM:
        //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
        //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        double s = Math.sqrt(6.0) / Math.sqrt(inputSize + outputSize);
        weightsMatrix =  Nd4j.rand(new int[] {outputSize, inputSize}, Nd4j.getDistributions().createUniform(-s, s));
        break;
      case XAVIER_LEGACY:
        weightsMatrix = Nd4j.randn(outputSize, inputSize, seed).div(FastMath.sqrt(outputSize + inputSize));
        break;
      case ZERO:
        weightsMatrix = Nd4j.zeros(outputSize, inputSize);
        break;
      case NORMAL:
      default:
        weightsMatrix = Nd4j.randn(outputSize, inputSize, seed);
    }
    LOG.debug("Weights initialized for outputSize {} and inputSize {}", outputSize, inputSize);
    LOG.trace("\n{}", weightsMatrix);
    return weightsMatrix;
  }

  public enum WeightInit {
    ZERO,
    NORMAL,
    XAVIER,
    XAVIER_LIKE,
    XAVIER_FAN_IN,
    XAVIER_UNIFORM,
    XAVIER_LEGACY
  }
}
