package fko.nnplayground;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LinearClassifierTest {

  private static final Logger LOG = LoggerFactory.getLogger(LinearClassifierTest.class);

  @Test
  void score() {

    LinearClassifier lc = new LinearClassifier(2, 2, 1, 3);
    lc.setUseRegularization(false);

    // data from
    // https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3
    // time: 46:00
    INDArray catWeights = Nd4j.create(new float[] {0.01f, -0.05f, 0.1f, 0.05f}, new int[] {4});
    INDArray carWeights = Nd4j.create(new float[] {0.7f, 0.2f, 0.05f, 0.16f}, new int[] {4});
    INDArray frogWeights = Nd4j.create(new float[] {0.0f, -0.45f, -0.2f, 0.03f}, new int[] {4});
    INDArray weightsMatrix = Nd4j.concat(0, catWeights, carWeights, frogWeights);

    INDArray b = Nd4j.create(new float[] {0.0f, 0.2f, -0.3f}, new int[] {3, 1});
    INDArray x_i = Nd4j.create(new float[] {-15f, 22f, -44f, 56f}, new int[] {4, 1});

    lc.setWeightsMatrix(weightsMatrix);
    lc.setBiasMatrix(b);

    INDArray scoreVector = lc.score(x_i);
    LOG.info("scoreVector {}", scoreVector);
    assertEquals("[-2.85,  0.86,  0.28]", scoreVector.toString());
  }

  @Test
  void softmaxScore() {

    /*
    cat   3.2   24.5    0.13  L=-log(0.13) = 0.89
    car   5.1   164.0   0.87
    frog -1.7   0.18    0.00
     */
    LinearClassifier lc = new LinearClassifier(2, 2, 1, 3);
    INDArray scoreMatrix = Nd4j.create(new float[] {3.2f, 5.1f, -1.7f}, new int[] {3, 1});

    INDArray p = lc.softmaxScore(scoreMatrix);
    LOG.info("scoreVector P softmax {}", p);
    assertEquals("[0.13,  0.87,  0.00]", p.toString());
  }

  @Test
  void softmax() {

    /*
    cat   3.2   24.5    0.13  L=-log(0.13) = 0.89
    car   5.1   164.0   0.87
    frog -1.7   0.18    0.00
     */

    // setup
    LinearClassifier lc = new LinearClassifier(2, 2, 1, 3);
    lc.setUseRegularization(false);
    INDArray x_i = Nd4j.create(new float[] {-15f, 22f, -44f, 56f}, new int[] {4, 1});


    // test 1
    INDArray scoreVector = Nd4j.create(new float[] {3.2f, 5.1f, -1.7f}, new int[] {3, 1});
    INDArray p = lc.softmaxScore(scoreVector);

    double loss = lc.softmaxLoss(x_i, p, 0);
    LOG.info("Loss Softmax 1 {}", loss);
    assertEquals(0.89, loss, 0.01);

    // data from
    // https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3
    // time: 46:00

    // test 2
    INDArray scoreVector2 = Nd4j.create(new float[] {-2.5f, 0.86f, 0.28f}, new int[] {3, 1});
    INDArray p2 = lc.softmaxScore(scoreVector2);

    double loss2 = lc.softmaxLoss(x_i, p2, 2);
    LOG.info("Loss Softmax 2 {}", loss2);
    assertEquals(0.452, loss2, 0.01);
  }

  @Test
  void lossSVM() {

    // data from
    // https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3
    // time: 46:00

    LinearClassifier lc = new LinearClassifier(2, 2, 1, 3);
    lc.setUseRegularization(false);
    INDArray x_i = Nd4j.create(new float[] {-15f, 22f, -44f, 56f}, new int[] {4, 1});

    INDArray scoreVector = Nd4j.create(new float[] {-2.5f, 0.86f, 0.28f}, new int[] {3, 1});

    double loss = lc.lossSVM(x_i, scoreVector, 2);
    LOG.info("Loss SVM {}", loss);
    assertEquals(1.58, loss, 0.01);
  }

  @Test
  void softmax_fullrun() {

    LinearClassifier lc = new LinearClassifier(2, 2, 1, 3);
    lc.setUseRegularization(false);

    // data from
    // https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3
    // time: 46:00
    INDArray catWeights = Nd4j.create(new float[] {0.01f, -0.05f, 0.1f, 0.05f}, new int[] {4});
    INDArray carWeights = Nd4j.create(new float[] {0.7f, 0.2f, 0.05f, 0.16f}, new int[] {4});
    INDArray frogWeights = Nd4j.create(new float[] {0.0f, -0.45f, -0.2f, 0.03f}, new int[] {4});
    INDArray weightsMatrix = Nd4j.concat(0, catWeights, carWeights, frogWeights);

    INDArray b = Nd4j.create(new float[] {0.0f, 0.2f, -0.3f}, new int[] {3, 1});
    INDArray x_i = Nd4j.create(new float[] {-15f, 22f, -44f, 56f}, new int[] {4, 1});

    lc.setWeightsMatrix(weightsMatrix);
    lc.setBiasMatrix(b);

    INDArray scoreVector = lc.score(x_i);
    LOG.info("scoreVector {}", scoreVector);
    assertEquals("[-2.85,  0.86,  0.28]", scoreVector.toString());

    // loss SVM
    double loss = lc.lossSVM(x_i, scoreVector, 2);
    LOG.info("Loss SVM {}", loss);
    assertEquals(1.58, loss, 0.01);

    // loss Softmax
    INDArray p = lc.softmaxScore(scoreVector);
    LOG.info("scoreVector P softmax {}", p);
    assertEquals("[0.02,  0.63,  0.35]", p.toString());

    double loss2 = lc.softmaxLoss(x_i, p, 2);
    LOG.info("Loss Softmax 2 {}", loss2);
    assertEquals(0.452, loss2, 0.01);

    lc.setLamda(0.1d);
    double rloss = lc.regularization();
    LOG.info("Regularization Loss {}", rloss);
    assertEquals(0.081, rloss, 0.001);

  }
}
