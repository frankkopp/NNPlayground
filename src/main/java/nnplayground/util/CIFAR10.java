package nnplayground.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * CIFAR10
 */
public class CIFAR10 {

  private static final Logger LOG = LoggerFactory.getLogger(CIFAR10.class);

  private static final String basePath = "./var/data" + "/cifar10";
  //private static final String dataUrl =  "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz";
  private static final String dataUrl =  "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

  public static void main(String[] args) throws Exception {

    LOG.info("Data load and vectorization using path {}", basePath);
    String localFilePath = basePath + "/cifar-10-binary.tar.gz";

    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
      LOG.info("Data downloaded from {}", dataUrl);
    }

    if (!new File(basePath + "/cifar-10-binary").exists()) {
      DataUtilities.extractTarGz(localFilePath, basePath);
    }

    LOG.info("Data extracted. Finished", basePath);
  }

}
