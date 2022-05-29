package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.checkerframework.checker.units.qual.A;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class objectDetectorClass {
    // hauria de començar amb lletra minúscula

    // s'utilitza per carregar el model i predir
    private Interpreter interpreter;
    // emmagatzemar totes les etiquetes en matriu
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // per RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // utilitzar per inicialitzar la gpu a l'aplicació
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;

    objectDetectorClass(AssetManager assetManager,String modelPath, String labelPath,int inputSize) throws IOException{
        INPUT_SIZE=inputSize;
        //utilitzar per definir gpu o CPU     no de fils
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); //configura-ho segons el teu telèfon
        // càrrega de model 
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        // carrega  labelmap
        labelList=loadLabelList(assetManager,labelPath);


    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        // per guardar l'etiqueta
        List<String> labelList=new ArrayList<>();
        // crear un nou lector
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        // recorre cada línia i emmagatzema-la a labelList
        while ((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // utilitzar per obtenir la descripció del fitxer
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // crear una nova funció Mat
    public Mat recognizeImage(Mat mat_image){
        // Gira la imatge original 90 graus per obtenir un marc retrat
        Mat rotated_mat_image=new Mat();
        Core.flip(mat_image.t(),rotated_mat_image,1);
        // si no feu aquest procés obtindreu una predicció incorrecta, menys no. d'objecte
        // ara el converteix en bitmap
        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        // defineix l'alçada i l'amplada
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        // escala el bitmap  a la mida d'entrada del model
         Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

         // convertir el bitmap a bytebuffer ja que l'entrada del model hi hauria d'estar
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

        //definició de sortida
        // 10: s'han detectat els 10 principals objectes
        // 4: hi ha coordenades a la imatge
        // flotant[][][]resultat=nou flotant[1][10][4];
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        // no utilitzarem aquest mètode de sortida
        // en lloc d'això, creem un treemap  de tres matrius (caixes, puntuació, classes)

        float[][][]boxes =new float[1][10][4];
        // 10: s'han detectat els 10 principals objectes
        // 4: hi ha coordenades a la imatge
        float[][] scores=new float[1][10];
        // emmagatzema puntuacions de 10 objectes
        float[][] classes=new float[1][10];
        // emmagatzema classe d'objecte

        // afegeix-ho a object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // ara predir
        interpreter.runForMultipleInputsOutputs(input,output_map);
        // Abans de veure aquest vídeo, si us plau, mireu els meus 2 vídeos anteriors
        // 1. Carregant el model tensorflow lite
        // 2. Objecte predictiu
        // En aquest vídeo dibuixarem caixes i l'etiquetarem amb el seu nom

        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        // recorre cada objecte
        // com a sortida només té 10 caixes
        for (int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);
            // definir el llindar de puntuació
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                // ho estem multiplicant amb l'alçada i l'amplada del marc originals

                float top=(float) Array.get(box1,0)*height;
                float left=(float) Array.get(box1,1)*width;
                float bottom=(float) Array.get(box1,2)*height;
                float right=(float) Array.get(box1,3)*width;
                //// dibuixa un rectangle al marc original // punt inicial // punt final de la caixa // color del gruix de la caixa
                Imgproc.rectangle(rotated_mat_image,new Point(left,top),new Point(right,bottom),new Scalar(0, 255, 0, 255),2);
                // escriure text al marc // cadena de nom de classe de l'objecte // punt de partida // color del text // mida del text
                Imgproc.putText(rotated_mat_image,labelList.get((int) class_value),new Point(left,top),3,1,new Scalar(255, 0, 0, 255),2);
            }

        }
        //seleccioneu el dispositiu i executeu-lo

        // abans de tornar, gireu enrere -90 graus
        Core.flip(rotated_mat_image.t(),mat_image,0);
        return mat_image;
        //Probar de rotar a 90*  appl123
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        //alguna entrada del model hauria de ser quant=0 per a un quant=1
        // per a aquest quant=0

        int quant=0;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // algun error
        //Ara corre

        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
    return byteBuffer;
    }
}