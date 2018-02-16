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

package fko.nnplayground.MinstNN;

import fko.nnplayground.UseCases.SimpleNeuralNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class SimpleNeuralNetworkTest {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleNeuralNetworkTest.class);

  @BeforeEach
  void setUp() {}

  @Test
  void constructor() {
    SimpleNeuralNetwork snn = new SimpleNeuralNetwork(28, 28, 1, 10, 100, 1234);
  }

  @Test
  void sigmoidTest() {
    INDArray array = Nd4j.ones(1, 1);
    array.put(0, 0, 1);
    INDArray sigmoidArray = Transforms.sigmoid(array);
    System.out.println("Array\n" + array);
    System.out.println("Sigmoid(array)\n" + sigmoidArray);
    INDArray derivedSigmoidArray = Transforms.sigmoidDerivative(sigmoidArray);
    System.out.println("Sigmoid'(array) lib\n" + derivedSigmoidArray);
    INDArray myDerivative = sigmoidArray.mul(-1).add(1).mul(sigmoidArray);
    System.out.println("Sigmoid'(array) manual \n" + myDerivative);

    // ==> bug in Nd4j sigmoidDerivative
  }

}

