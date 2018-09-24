package org.proyecto2018.jcf.wiflow;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    private final String TAG = "TAG_WIFLOW";
    private TensorFlowInferenceInterface inferenceInterface;
    private float[] output;
    private TextView salida;
    private Button buttonScan;
    private String texto = "";
    private static WifiManager wifiMng;
    private static WiFiScanReceiver wifiReceiver;
    private String[] macs = {
            "c0:c1:c0:b0:3a:9b",
            "08:7a:4c:34:a0:24",
            "f8:35:dd:6e:50:29",
            "00:13:10:93:be:f8",
            "d8:d7:75:ee:23:26",
            "fa:8f:ca:54:7a:29",
            "70:4f:57:82:4a:1c",
            "44:40:b0:5d:e7:15",
            "84:17:ef:4e:92:e0",
            "5c:f4:ab:bf:38:40",
            "c8:3d:d4:e4:9a:a0",
            "84:00:2d:2c:ad:85",
            "24:7f:20:08:d1:36",
            "a4:08:f5:f0:ee:72",
            "b4:75:0e:d9:7b:fb",
            "c8:3d:d4:d9:12:40",
            "50:6a:03:c8:db:2d",
            "7c:b7:33:29:cb:17",
            "e4:be:ed:c0:6b:48",
            "98:de:d0:42:fe:46"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        salida = (TextView) findViewById(R.id.salida);
        buttonScan = (Button) findViewById(R.id.buttonScan);


        // ejercitar el modelo con varios condiciones de entrada
        float[][] atributos = {
                {50,48,45,44,31,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {49,45,35,44,35,0,34,34,30,0,0,0,0,0,0,0,0,0,0,0},
                {36,45,44,40,32,0,0,36,0,0,0,0,0,0,0,0,0,0,0,0},
                {30,49,62,37,0,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {30,45,64,36,32,0,0,33,0,0,0,0,0,0,0,0,0,0,0,0},
                {0,53,58,35,0,0,0,34,30,0,0,0,0,0,0,0,0,0,0,0},
                {31,42,46,35,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {48,49,46,49,0,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {44,44,49,0,0,0,0,37,0,0,0,0,0,0,0,0,0,0,0,0},
                {48,54,55,48,0,30,0,43,0,0,0,0,0,0,0,0,0,0,0,0},
                {40,53,58,50,0,0,0,37,0,0,0,0,0,0,0,0,0,0,0,0},
                {44,56,64,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {33,56,52,44,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0},
                {43,47,52,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {49,50,49,51,0,33,0,40,0,0,0,0,0,0,0,0,0,0,0,0},
                {46,42,51,44,0,0,0,33,0,0,0,0,0,0,0,0,0,0,0,0},
                {33,51,55,43,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {44,52,61,50,0,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {40,49,54,43,0,0,0,36,0,0,0,0,0,0,0,0,0,0,0,0},
                {47,58,61,44,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0},
                {32,40,44,38,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0},
                {37,41,48,55,0,30,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {39,41,52,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {42,45,57,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {44,46,53,44,0,0,0,33,0,0,0,0,0,0,0,0,0,0,0,0},
                {49,39,48,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {40,49,50,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {39,59,52,46,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0},
                {57,41,47,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {59,34,53,45,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {56,45,52,57,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {49,44,38,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {47,38,0,45,0,0,0,34,0,0,0,0,31,0,0,0,0,0,0,0},
                {47,33,0,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {47,0,0,46,30,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {51,37,32,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {52,30,34,51,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0},
                {59,36,47,54,0,32,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {50,0,36,48,0,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {43,39,0,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {47,38,35,53,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {47,35,34,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {50,40,35,46,0,0,0,30,0,30,0,0,0,0,0,0,0,0,0,0},
                {55,35,40,56,0,0,0,0,0,30,0,0,0,0,0,0,0,0,0,0},
                {46,39,35,40,0,0,0,0,0,0,0,0,0,32,0,0,30,0,0,0},
                {50,35,37,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {45,34,36,51,0,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {50,33,43,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {55,38,43,52,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0},
                {55,39,44,44,0,0,0,33,0,0,0,0,0,0,0,0,0,0,0,0},
                {62,41,51,49,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0},
                {62,47,37,46,0,0,0,34,0,0,0,0,0,0,0,30,0,0,0,0},
                {65,54,45,55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {67,58,43,54,0,0,0,0,0,0,0,0,0,0,0,31,0,0,0,0},
                {54,33,0,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {53,37,31,44,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0},
                {59,37,32,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {50,42,44,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {48,34,45,52,0,0,0,31,0,0,0,0,0,0,0,0,0,0,0,0},
                {51,44,44,44,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0},
                {53,38,39,51,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0},
                {60,51,39,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {63,55,44,47,0,0,0,0,0,0,0,0,0,31,0,0,0,0,0,0},
                {69,44,37,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {70,52,41,51,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0},
                {36,0,0,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {53,31,30,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {54,39,32,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {56,40,0,55,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0},
                {53,43,41,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {55,36,0,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {49,37,0,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {64,56,46,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {72,45,39,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
        };

        double[][] etiquetas = {
                {0.9,7.9},
                {0.9,8.9},
                {0.9,9.9},
                {0.9,10.9},
                {0.9,11.9},
                {0.9,12.9},
                {0.9,13.9},
                {2.8,7.9},
                {2.8,8.9},
                {2.8,9.9},
                {2.8,10.9},
                {2.8,11.9},
                {2.8,12.9},
                {2.8,13.9},
                {3.8,7.9},
                {3.8,8.9},
                {3.8,9.9},
                {3.8,10.9},
                {3.8,11.9},
                {3.8,12.9},
                {3.8,13.9},
                {4.8,7.9},
                {4.8,8.9},
                {4.8,9.9},
                {4.8,10.9},
                {4.8,11.9},
                {4.8,12.9},
                {4.8,13.9},
                {6.4,7.1},
                {6.4,8.1},
                {6.4,9.1},
                {6.4,10.1},
                {6.4,1.0},
                {6.4,2.0},
                {6.4,3.0},
                {6.4,4.0},
                {6.4,5.0},
                {6.4,6.0},
                {7.4,1.0},
                {7.4,2.0},
                {7.4,3.0},
                {7.4,4.0},
                {7.4,5.0},
                {7.4,6.0},
                {8.4,1.0},
                {8.4,2.0},
                {8.4,3.0},
                {8.4,4.0},
                {8.4,5.0},
                {8.4,6.0},
                {8.4,7.0},
                {8.4,8.0},
                {8.4,9.0},
                {8.4,10.0},
                {9.4,1.0},
                {9.4,2.0},
                {9.4,3.0},
                {9.4,4.0},
                {9.4,5.0},
                {9.4,6.0},
                {9.4,7.0},
                {9.4,8.0},
                {9.4,9.0},
                {9.4,10.0},
                {10.4,1.0},
                {10.4,2.0},
                {10.4,3.0},
                {10.4,4.0},
                {10.4,5.0},
                {10.4,6.0},
                {10.4,7.0},
                {10.4,8.0},
                {10.4,9.0},
                {10.4,10.0}
        };

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "model_final.hdf5.pb");
        for(int i=0 ; i<atributos.length ; i++){
            output = predict(atributos[i]);
            Log.d(TAG, "Predicción: " + String.format("%.2f", output[0]) + ", " + String.format("%.2f", output[1])
            + " Posición real: " + String.format("%.2f", etiquetas[i][0]) + ", " + String.format("%.2f", etiquetas[i][1]));
            texto += "Predicción: " + String.format("%.2f", output[0]) + ", " + String.format("%.2f", output[1])
                    + " Posición real: " + String.format("%.2f", etiquetas[i][0]) + ", "
                    + String.format("%.2f", etiquetas[i][1]) + "\n";

            salida.setText(texto);
        }

        RegistrarReceptorWiFi();

        buttonScan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(wifiMng.startScan() == true) {
                    salida.setText("\n\n\n" + "scan iniciado");
                } else {
                    salida.setText("\n\n\n" + "Falló inicio de scan");
                }
            }
        });

    }

    private float[] predict(float[] input){
        // nuestro modelo tiene dos neuronas de salida
        float output[] = new float[2];

        inferenceInterface.feed("dense_input", input, 1, input.length);
        inferenceInterface.run(new String[]{"dense_4/Relu"});
        inferenceInterface.fetch("dense_4/Relu", output);

        return output;
    }

    private void RegistrarReceptorWiFi()
    {
        wifiReceiver = new WiFiScanReceiver();

        if (wifiReceiver != null)
        {
            registerReceiver(wifiReceiver,
                    new IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION));
            wifiMng = (WifiManager) getApplicationContext().getSystemService(this.WIFI_SERVICE);
            wifiMng.setWifiEnabled(true);
        }
    }

    class WiFiScanReceiver extends BroadcastReceiver {

        private static final String TAG = "JCDBG";
        ScanResult scanres;
        String macAdd = "";
        int nivel = 0;
        List<ScanResult> wifiScanList;
        float[] atributosNivel = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        @Override
        public void onReceive(Context context, Intent intent)
        {
            Log.d(TAG, "WiFiScanReceiver onReceive");
            salida.setText("WiFiScanReceiver onReceive");
            wifiScanList = wifiMng.getScanResults();
            for(int i = 0; i < wifiScanList.size(); i++) {
                scanres = wifiScanList.get(i);
                macAdd = scanres.BSSID;
                nivel = scanres.level;

                for(int j = 0; j < macs.length; j++) {
                    if(macs[j].equals(macAdd)) {
                        atributosNivel[j] = nivel+100; // sumo 100 de acuerdo a como se entrenó el modelo
                    }
                }
            }

            output = predict(atributosNivel);
            Log.d(TAG, "Predicción: " + String.format("%.2f", output[0]) + ", " + String.format("%.2f", output[1]));
            texto = "Predicción: x=" + String.format("%.2f", output[0]) + ", y=" + String.format("%.2f", output[1]);
            salida.setText("\n\n\n" + texto);
            if(wifiMng.startScan() == true) {
                Log.i(TAG, "WiFiScanReceiver: Scan iniciado");
            } else {
                Log.i(TAG, "WiFiScanReceiver: Falló inicio de scan");
                salida.setText("\n\n\n" + "Falló inicio automático de scan");
            }
        }
    }
}
