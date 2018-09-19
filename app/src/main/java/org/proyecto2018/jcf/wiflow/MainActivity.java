package org.proyecto2018.jcf.wiflow;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    private final String TAG = "TAG_WIFLOW";
    private TensorFlowInferenceInterface inferenceInterface;
    private float[] output;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "model_wiflow.pb");
        // run the model for all possible inputs i.e. [0,0], [0,1], [1,0], [1,1]
        // ejercitar el modelo con varios condiciones de entrada
        //float[] input = {50,47,45,44,31,30,30,0,0,0,0,0,0,0,0,0,0,0,0,0}; // 0.9, 7.9
        //float[] input = {49,34,35,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // 6.4, 3.0
        //float[] input = {38,52,57,41,30,0,0,37,30,0,0,0,0,0,0,0,0,0,0,0}; // 0.9, 12.9
        float[][] atributos = {
                {50,47,45,44,31,30,30,0,0,0,0,0,0,0,0,0,0,0,0,0}, // 0.9, 7.9
                {49,34,35,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // 6.4, 3.0
                {38,52,57,41,30,0,0,37,30,0,0,0,0,0,0,0,0,0,0,0} // 0.9, 12.9
        };

        for(int i=0 ; i<atributos.length ; i++){
            output = predict(atributos[i]);
            Log.d(TAG, "Salida: " + String.format("%.2f", output[0]) + ", " + String.format("%.2f", output[1]));
        }
    }

    private float[] predict(float[] input){
        // model_wiflow tiene dos neuronas de salida
        float output[] = new float[2];

        inferenceInterface.feed("dense_input", input, 1, input.length);
        inferenceInterface.run(new String[]{"dense_4/Relu"});
        inferenceInterface.fetch("dense_4/Relu", output);

        return output;
    }
}
