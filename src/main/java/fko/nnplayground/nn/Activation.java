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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * Activation
 */
public class Activation {

  private static final Logger LOG = LoggerFactory.getLogger(Activation.class);

  private static double leakyReLUalpha = 1e-2d;

  public static INDArray applyActivation(final Activations fun, final INDArray in) {
    final INDArray out;
    switch (fun) {
      case IDENTITY:
        out = Transforms.identity(in);
        break;
      case SIGMOID:
        //out = Transforms.sigmoid(in);
        out = Transforms.pow(Transforms.exp(in.mul(-1)).add(1), -1);
        break;
      case TANH:
        //out = Transforms.tanh(in);
        out = Transforms.pow(Transforms.exp(in.mul(-2)).add(1), -1).mul(2).sub(1);
        break;
      case RELU:
        //out = Transforms.relu(in);
        out = Transforms.max(in, 0);
        break;
      case LEAKYRELU:
        //out = Transforms.leakyRelu(in);
        out = in.dup();
        BooleanIndexing.applyWhere(out, Conditions.lessThanOrEqual(0),
                input -> leakyReLUalpha * input.doubleValue());
        break;
      case SOFTMAX:
        // Exercise to calculate this manually instead of library call
        // it's a bit faster as well :)

        // compute class probabilities
        // shift the values of f for each example so that the highest number is 0:
        // otherwise e^sk might become a NaN (too high)
        final INDArray s = in.subRowVector(in.max(0));
        // s = f(x,W) ==> P = e^sk / sum_j(e^sj);
        // get un-normalized probabilities
        final INDArray es = exp(s);
        // normalize them for each(!) example - each col sums to one
        out = es.divRowVector(es.sum(0));

        // softmax is a row operation so we need to transpose our
        // matrix before and the result after
        // output = Transforms.softmax(output.transpose()).transpose();
        // hiddenOutput = Transforms.softmax(hiddenOutput);
        break;
      default:
        out = Transforms.identity(in);
    }
    LOG.trace("Activation of \n{} is \n{}", in.ravel(), out.ravel() );
    return out;
  }

  public static INDArray applyDerivative(final Activations fun, final INDArray in) {
    final INDArray dIn;
    switch (fun) {
      case IDENTITY:
        // f(x) = x ==> f'(x) = 1
        dIn = in.assign(1);
        break;
      case SIGMOID:
        // dIn = Transforms.sigmoidDerivative(in); // Library is buggy
        dIn = in.mul(-1).add(1).mul(in);
        break;
      case TANH:
        // f'(x) = 1 - f(x)^2
        dIn = in.mul(in).mul(-1).add(1);
        break;
      case RELU:
        // ReLU(x) = max(0,x) ==> ReLU'(x) = 0 for <=0 OR 1 for >0
        dIn = in.dup();
        BooleanIndexing.replaceWhere(dIn, 0, Conditions.lessThanOrEqual(0));
        BooleanIndexing.replaceWhere(dIn, 1, Conditions.greaterThan(0));
        break;
      case LEAKYRELU:
        dIn = Transforms.leakyReluDerivative(in, 0);
        break;
//      case SOFTMAX: // only useful in output layer
      // get the correct label matrix and transpose it - this creates a matrix
      // identical in shape to scores with 1 where the correct score should have been
      // pk−1(yi=k)
      // the nonLinDerivative is identity pk except for all scores at the correct label there it is pk-1
      //dIn = in.sub(labels.transpose()).div(features.columns());
//        break;
      default:
        // f(x) = x ==> f'(x) = 1
        dIn = in.assign(1);
        break;
    }
    LOG.trace("Derivative of \n{} is \n{}", in.ravel(), dIn.ravel() );
    return dIn;
  }

  public static double getLeakyReLUalpha() {
    return leakyReLUalpha;
  }

  public static void setLeakyReLUalpha(final double leakyReLUalpha) {
    Activation.leakyReLUalpha = leakyReLUalpha;
  }

  /**
   * Transfer Functions / Activation Functions
   */
  public enum Activations {

    /**
     * no op
     * f(x) = x
     */
    IDENTITY,

    /**
     * Sigmoid or Logistic Activation Function
     * Squish the output between 0 and 1
     * 1 / (1 + e^(-activation))
     */
    SIGMOID,

    /**
     * Tanh or hyperbolic tangent Activation Function
     * Squish the output between -1 and 1
     * 1 - (2 / (e^2x - 1))
     */
    TANH,

    /**
     * ReLU (Rectified Linear Unit) Activation Function
     * 1 / (1 + e^-x))
     */
    RELU,

    /**
     * Leaky ReLU
     * <0: f(x) = ax (a usually 0.01
     * >0: f(x) = x
     */
    LEAKYRELU,

    /**
     * Softmax
     */
    SOFTMAX


  }
}
