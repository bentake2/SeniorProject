package com.example.senior_project_image_sementation;

import com.example.senior_project_image_sementation.ModelClasses;
import com.example.senior_project_image_sementation.R;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.LiteModuleLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;

    private static final int BACKGROUND = 0;
    private static final int AEROPLANE = 1;
    private static final int BICYCLE = 2;
    private static final int BIRD = 3;
    private static final int BOAT = 4;
    private static final int BOTTLE = 5;
    private static final int BUS = 6;
    private static final int CAR = 7;
    private static final int CAT = 8;
    private static final int CHAIR = 9;
    private static final int COW = 10;
    private static final int DININGTABLE = 11;
    private static final int DOG = 12;
    private static final int HORSE = 13;
    private static final int MOTORBIKE = 14;
    private static final int PERSON = 15;
    private static final int POTTEDPLANT = 16;
    private static final int SHEEP = 17;
    private static final int SOFA = 18;
    private static final int TRAIN = 19;
    private static final int TVMONITOR = 20;
    private static final int CLASSNUM = 21;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button segmentButton = (Button) findViewById(R.id.segment);
        ProgressBar progressBar = (ProgressBar) findViewById(R.id.progressBar);


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        buttonLoadImage.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i, RESULT_LOAD_IMAGE);
                progressBar.setVisibility(ProgressBar.VISIBLE);

            }
        });
        //progressBar = (ProgressBar) findViewById(R.id.progressBar);
        segmentButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {

                Bitmap bitmap = null;
                Module module = null;



                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);


                try {
                    //Read the image as Bitmap
                    bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();

                    //Here we reshape the image into 400*400
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

                    //Loading the model file.
                    module = LiteModuleLoader.load(MainActivity.fetchModelFile(getApplicationContext(), "deeplabv3_scripted_optimized.ptl"));
                } catch (IOException e) {
                    finish();
                }

                //Input Tensor
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                final float[] inputs = inputTensor.getDataAsFloatArray();

                Map<String, IValue> outTensors =
                        module.forward(IValue.from(inputTensor)).toDictStringKey();

                // the key "out" of the output tensor contains the semantic masks
                // see https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101
                final Tensor outputTensor = outTensors.get("out").toTensor();
                final float[] outputs = outputTensor.getDataAsFloatArray();

                int width = bitmap.getWidth();
                int height = bitmap.getHeight();

                int[] intValues = new int[width * height];

                // go through each element in the output of size [WIDTH, HEIGHT] and
                // set different color for different classnum
                for (int j = 0; j < width; j++) {
                    for (int k = 0; k < height; k++) {
                        // maxi: the index of the 21 CLASSNUM with the max probability
                        int maxi = 0, maxj = 0, maxk = 0;
                        double maxnum = -100000.0;
                        for (int i=0; i < CLASSNUM; i++) {
                            if (outputs[i*(width*height) + j*width + k] > maxnum) {
                                maxnum = outputs[i*(width*height) + j*width + k];
                                maxi = i; maxj = j; maxk= k;
                            }
                        }
                        // color coding for person (red), dog (green), sheep (blue)
                        // black color for background and other classes

                        if (maxi == AEROPLANE)
                            intValues[maxj*width + maxk] = 0xFFFF0000; // red
                        else if (maxi == BICYCLE)
                            intValues[maxj*width + maxk] = 0xFFFF69B4; // HotPink
                        else if (maxi == BIRD)
                            intValues[maxj*width + maxk] = 0xFFFF7F50; // Coral
                        else if (maxi == BOAT)
                            intValues[maxj*width + maxk] = 0xFFFFFF00; // Yellow
                        else if (maxi == BOTTLE)
                            intValues[maxj*width + maxk] = 0xFFE6E6FA; // Lavender
                        else if (maxi == BUS)
                            intValues[maxj*width + maxk] = 0xFFADFF2F; // GreenYellow
                        else if (maxi == CAR)
                            intValues[maxj*width + maxk] = 0xFF00FFFF; // Aqua
                        else if (maxi == CAT)
                            intValues[maxj*width + maxk] = 0xFFFFE4C4; // Bisque
                        else if (maxi == CHAIR)
                            intValues[maxj*width + maxk] = 0xFFF0FFF0; // HoneyDew
                        else if (maxi == COW)
                            intValues[maxj*width + maxk] = 0xFFD3D3D3; // LightGray
                        else if (maxi == DININGTABLE)
                            intValues[maxj*width + maxk] = 0xFF8B0000; // DarkRed
                        else if (maxi == DOG)
                            intValues[maxj*width + maxk] = 0xFFC71585; // MediumVioletRed
                        else if (maxi == HORSE)
                            intValues[maxj*width + maxk] = 0xFF0000FF; // DarkOrange
                        else if (maxi == MOTORBIKE)
                            intValues[maxj*width + maxk] = 0xFFBDB76B; // DarkKhaki
                        else if (maxi == PERSON)
                            intValues[maxj*width + maxk] = 0xFFFF00FF; // Magenta
                        else if (maxi == POTTEDPLANT)
                            intValues[maxj*width + maxk] = 0xFF32CD32; // LimeGreen
                        else if (maxi == SHEEP)
                            intValues[maxj*width + maxk] = 0xFF4682B4; // SteelBlue
                        else if (maxi == SOFA)
                            intValues[maxj*width + maxk] = 0xFF8B4513; // SaddleBrown
                        else if (maxi == TRAIN)
                            intValues[maxj*width + maxk] = 0xFF191970; // MidnightBlue
                        else if (maxi == TVMONITOR)
                            intValues[maxj*width + maxk] = 0xFF2F4F4F; // DarkSlateGray
                        else
                            intValues[maxj*width + maxk] = 0xFF000000; // black


                    }
                }

                Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
                Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
                outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0,
                        outputBitmap.getWidth(), outputBitmap.getHeight());
                imageView.setImageBitmap(outputBitmap);
                progressBar.setVisibility(ProgressBar.INVISIBLE);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);


        }


    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}