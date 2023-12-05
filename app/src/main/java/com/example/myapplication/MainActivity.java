package com.example.myapplication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.Manifest;
import android.widget.TextView;
import com.example.myapplication.ml.Model;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    //Size of image for model
    int imageInModelSize =  224;

    Button button;
    ImageView imageImageView;
    TextView classTextView;
    TextView probabilityTextView;

    String[] classesNames = {
            "Banana",
            "Onion",
            "Zucchini",
            "EmptyDesk"
    };

    //Oblicza index najwięskzej wagi
    public int findIndexOfMin(float[] array) {
        // Check if the array is empty or null to avoid errors
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }

        // Initialize variables to hold the smallest value and its index
        float min = array[0];
        int minIndex = 0;

        // Loop through the array to find the smallest element and its index
        for (int i = 1; i < array.length; i++) {
            if (array[i] > min) {
                min = array[i];
                minIndex = i;
            }
        }

        // Return the index of the smallest value found
        return minIndex;
    }

    public float findMin(float[] array) {
        // Check if the array is empty or null to avoid errors
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }

        // Initialize variables to hold the smallest value and its index
        float min = array[0];

        // Loop through the array to find the smallest element and its index
        for (int i = 1; i < array.length; i++) {
            if (array[i] > min) {
                min = array[i];
            }
        }

        // Return the index of the smallest value found
        return min;
    }

    //Inserts pixesls from image to byteBuffer
    //ByteBuffer is a pointer so it operates on origilan
    public void insertBytesFromImage(Bitmap image, ByteBuffer byteBuffer)
    {
        //Przechodzi przez zdjęcie i wypełnia byteBuffer wartośćiami pixeli
        for (int y = 0; y < imageInModelSize; y++) {
            for (int x = 0; x < imageInModelSize; x++) {
                int pixelValue = image.getPixel(x, y);
                //Przesuwanie wartości dla R G i B tak by można je było w sposó rozróżnialny zapisać w jednym przedziale <0,1>
                byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f); // R
                byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);  // G
                byteBuffer.putFloat((pixelValue & 0xFF) / 255.0f);        // B
            }
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Łączy obiekt z kodu z obiektem w xml
        button = findViewById(R.id.button);
        imageImageView = findViewById(R.id.image);
        classTextView = findViewById(R.id.classes);
        probabilityTextView = findViewById(R.id.probability);

        //Dodaje nasłuch na eventy
        button.setOnClickListener(new View.OnClickListener() {
            //Czeka na wciśnięcie przycisku
            @Override
            public void onClick(View v) {
                //Najpierw sprawdz czy jest w ogóle pozwolenia na dostęp do kamery
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                {
                    //Otwiera kamerę w trybie zrobienia jednego zdjęcia
                    Intent camera = new Intent((MediaStore.ACTION_IMAGE_CAPTURE));
                    //Włącza wyżej stworzoną katywność i oczekuje na jej wynik, bez "ForResult" po zrobieniu zjęcia wyszło by z aplikacji, numer wybiera rodzaj aktywaności
                    startActivityForResult(camera, 1);
                }else
                {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image) throws IOException {

        //Takes context of whole application
        Model model = Model.newInstance(getApplicationContext());
        // Skaluje zdjęcie tak by miało wielkość obsługiwaną przez model
        image = Bitmap.createScaledBitmap(image, imageInModelSize, imageInModelSize, true);

        // Creates inputs for reference. 1 - batchSize, 3 - RGB
        TensorBuffer inputFeature = TensorBuffer.createFixedSize(new int[] {1, imageInModelSize, imageInModelSize, 3}, DataType.FLOAT32);

        //Wielkość ByteBffer'a szerokośćZdjęcia * wysokośćZdjęcia * RGH * float32Size
        int bufferSize = imageInModelSize * imageInModelSize * 3 * 4;
        //Tworzy BytBuffer z wyzerowaną zawarrtością
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        //Ustawia  kolejność bajtów w pamięci zgodznie ze standardem obecnego hardware'u
        byteBuffer.order(ByteOrder.nativeOrder());

        insertBytesFromImage(image,byteBuffer);

        //Konwertuje przetworzone zdjęcie na staworzny tensor będącym klasą dostarczoną przez tensorflow, którzy przesyłany jest do modelu
        inputFeature.loadBuffer(byteBuffer);

        //Przetworznie danych przez model i zwrot wyników
        Model.Outputs outputs = model.process(inputFeature);
        //Wyciąga wartiości jako tensor
        TensorBuffer outputFeature = outputs.getOutputFeature0AsTensorBuffer();
        model.close();

        int maxValueIndex = findIndexOfMin(outputFeature.getFloatArray());
        //Ustawia orazpoznaną wartość
        classTextView.setText(classesNames[maxValueIndex]);
        probabilityTextView.setText(String.valueOf(findMin(outputFeature.getFloatArray())));
    }

    //Uruchamiana automatycznie po zakończeniu aktywności w tym przypadku crozmiania zdjęcia
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        //Sprawdza czy aktywność została zakończona poprawnia
        if(resultCode == RESULT_OK)
        {
            //getExtras() pobiera wszystkie elmenty dodane do "data" a następnie pobiera tą o nazwi "data"
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int imageDimension = Math.min(image.getWidth(), image.getHeight());
            //Dostosowywuje zdjęcie do wymiarów
            image = ThumbnailUtils.extractThumbnail(image, imageInModelSize, imageInModelSize);
            //Wyświatla zdjęci w aplikacji
            imageImageView.setImageBitmap(image);

            try {
                classifyImage(image);
            }catch (IOException e) {
                Log.d("tag", "IOException");
            }
        }
    }
}