package fko.nnplayground.MinstNN;

import org.apache.commons.math3.util.FastMath;
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

