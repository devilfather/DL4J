package Modelling;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class CSVDataModel2 {
    private static MultiLayerNetwork model;
    public static String dataLocalPath;
    private static DataSet trainingData;
    private static DataSet testData;
    private static org.deeplearning4j.optimize.api.IterationListener iterationListener;

    public static void main(String[] args) throws Exception {
        int batchSize = 768;
        int seed = 12345;
        int numInputs = 8;
        int numOutputs = 1;
        int nEpochs = 5000;

        System.out.println("................Import data..................");

        dataLocalPath = "C:\\Users\\jiayi\\Desktop\\dl4j-examples\\untitled\\src\\main\\resources\\";
        String filename = new File(dataLocalPath, "pima-indians-diabetes.csv").getAbsolutePath();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filename)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, 8, 8, true);
        DataSet allData = iterator.next();

        System.out.println("................Split Data..................");
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.85);  //Use 80% of data for training

        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);
        System.out.println(trainingData);
        System.out.println("................Data Loading Complete..................");

        System.out.println("................Build Neural Network Configuration..................");
        //Data Setting Complete... built a train neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(8)
                .build()
            )
            .layer(1, new DenseLayer.Builder().nIn(8).nOut(4)
                .build()
            )
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                .activation(Activation.RELU)
                .nIn(4).nOut(numOutputs).build()
            )
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        System.out.println(model.summary());

        System.out.println("................Training..................");
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainingData);
        }

         System.out.println("................Evaluate..................");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(testData.getFeatures());
        INDArray lables = testData.getLabels();
        eval.eval(lables,output);
        System.out.println(eval.stats());
        System.out.println("................Done..................");
    }
}
