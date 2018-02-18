package fko.nnplayground.nn;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class ActivationTest {

  private static final Logger LOG = LoggerFactory.getLogger(ActivationTest.class);

  @Test
  public void testDerivatives() {

    int seed = 123;

    INDArray y = Nd4j.zeros(2, 1);
    y.put(0, 0, 1);
    LOG.debug("Labels y=\n{}", y);

    INDArray x = Nd4j.randn(2, 1, seed);
    LOG.debug("Input x=\n{}", x);

    INDArray softmax = Activation.applyActivation(Activation.Activations.SOFTMAX, x);
    LOG.debug("Softmax of x=\n{}", softmax);

    INDArray dSoftmax = Activation.applyDerivative(Activation.Activations.SOFTMAX, softmax);
            // x.sub(y.transpose()).div(1);
    LOG.debug("dSoftmax of x=\n{}", dSoftmax);
  }

}