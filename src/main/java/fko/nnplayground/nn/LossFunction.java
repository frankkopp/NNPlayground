package fko.nnplayground.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LossFunction {

  public static double computeLoss(LossFunction.LossFunctions fun, INDArray labels, INDArray activations,
                                   int nExamples) {
    switch (fun) {
      case MSE:
        // quadratic cost function
        // (1/2n) * ∑ ||y(x)-aL(x)||^2
        return Transforms.pow(labels.sub(activations), 2).sumNumber().doubleValue() / (2 * nExamples);
      default:
        // TODO Exception
        return 0;
    }
  }

  public static INDArray computeLossGradient(LossFunction.LossFunctions fun, INDArray labels, INDArray activations,
                                             int nExamples) {
    switch (fun) {
      case MSE:
        // this is the gradient/derivative of the quadratic cost function!
        // (aL−y) - the derivative of cost also has a 1/n - this is done in the weights
        // update to also cover the Bias
        return activations.sub(labels);
      default:
        // TODO Exception
        return null;
    }
  }

  public enum LossFunctions {

    /**
     * Mean Squared Error (MSE), or quadratic, loss function
     * (1/2n) * ∑ ||y(x)-aL(x)||^2
     */
    MSE

  }
}
