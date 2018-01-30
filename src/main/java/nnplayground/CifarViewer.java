package nnplayground;

import com.sun.javafx.application.PlatformImpl;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** CifarViewer */
public class CifarViewer extends Application {

  private static final Logger LOG = LoggerFactory.getLogger(CifarViewer.class);

  private Stage primaryStage;
  private final FlowPane flowPane = new FlowPane();

  private final int width = 32;
  private final int height = 32;
  private int zoom = 4;

  public CifarViewer() {

    // Startup the JavaFX platform
    Platform.setImplicitExit(false);
    PlatformImpl.startup(
        () -> {
          primaryStage = new Stage();
          start(primaryStage);
        });

    // wait for the UI to show before returning
    do {
      try {
        Thread.sleep(100);
      } catch (InterruptedException ignore) {
      }
    } while (primaryStage == null || !primaryStage.isShowing());

  }

  @Override
  public void start(final Stage stage) {

    // raw score out put row
    ScrollPane scrollPane = new ScrollPane(flowPane);
    scrollPane.setFitToWidth(true);

    // horizontal box with all rows
    VBox root = new VBox();
    root.setAlignment(Pos.TOP_LEFT);
    root.getChildren().addAll(scrollPane);

    Scene scene = new Scene(root, 520, 300);
    stage.setScene(scene);
    stage.setTitle("Cifar Viewer");
    stage.setResizable(true);
    stage.show();
  }

  public void showImage(final DataSet dataSet) {

    Image cifarImage = createImage(dataSet);

    ImageView imageView = new ImageView(cifarImage);
    imageView.setFitHeight(zoom * height);
    imageView.setFitWidth (zoom * width);

    Platform.runLater(() -> flowPane.getChildren().add(imageView));
  }

  @NotNull
  private Image createImage(final DataSet dataSet) {

    WritableImage cifarImage = new WritableImage(width, height);

    int w = (int) cifarImage.getWidth();
    int h = (int) cifarImage.getHeight();

    PixelWriter writer = cifarImage.getPixelWriter();

    INDArray dsRow = dataSet.getFeatures();

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        double red = dsRow.getDouble(0,0,j,i);
        double green = dsRow.getDouble(0,1,j,i);
        double blue = dsRow.getDouble(0,2,j,i);
        Color c = new Color(red, green, blue, 1.0);
        writer.setColor(i, j, c);
      }
    }
    return cifarImage;
  }
}
