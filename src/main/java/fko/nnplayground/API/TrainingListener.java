package fko.nnplayground.API;


import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public interface TrainingListener {

  void iterationDone(final Network nn, final int iteration);

  void onEpochStart(final Network nn);
  
  void onEpochEnd(final Network nn);

  void onForwardPass(final Network nn, final List<INDArray> activations);

  void onForwardPass(final Network nn, final Map<String, INDArray> activations);

  void onGradientCalculation(final Network nn);
  
  void onBackwardPass(final Network nn);
  
}
