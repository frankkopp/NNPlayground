package fko.nnplayground.MinstNN;

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

  private final INDArray labels;

  private INDArray error;

  private double totalError;


  public OutputLayer(final int inputSize, final int outputSize, final INDArray labels, final WeightInitializer.WeightInit weightInit, final Activations activationFunction, final int seed) {
    super(inputSize, outputSize, weightInit, activationFunction, seed);
    this.labels = labels;
  }

  @Override
  public INDArray forwardPass(final INDArray outputLastLayer, final boolean computeError) {
    super.forwardPass(outputLastLayer);
    if (computeError) {
      computeError(computeError);
      computeTotalError(computeError);
    }
    return getActivation();
  }

  @Override
  public INDArray backwardPass() {
    super.backwardPass(error);
    return getPreviousLayerError();
  }

  @Override
  public INDArray getLabels() {
    return labels;
  }

  @Override
  public INDArray computeError(final boolean training) {
    error = getActivation().sub(labels);
    return error;
  }

  @Override
  public double computeTotalError(final boolean training) {
    totalError = Transforms.abs(error).meanNumber().doubleValue();
    return totalError;
  }

  @Override
  public INDArray getError() {
    return error;
  }

  @Override
  public double getTotalError() {
    return totalError;
  }
}
