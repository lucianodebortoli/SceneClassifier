package com.audio.sceneclassifier;

//           LUCIANO DE BORTOLI
//          INGENIERIA EN SONIDO
// UNIVERSIDAD NACIONAL DE TRES DE FEBRERO
//              2018 - 2019
//
//       REQUIRED CUSTOM CLASSES:
//          -InformationActivity.java
//          -MelSpectrogram.java
//          -SpectrogramView.java

import android.Manifest;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Locale;
import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {


    //------------------------------------- UI OBJECTS ---------------------------------------------

    TextView predictionView_0, predictionView_1, predictionView_2, languageTextView,
             bufferTextView, bufferEditTextView, dynamicRangeTextViewEdit, dynamicRangeTextView,
            xLabelTextView, dataSetTextView, colorMapTextView, integrationTextView;
    Button logButton, validateButton, runButton, infoButton;
    ProgressBar predictionBar1, predictionBar2, predictionBar3, loadingBar, horizontalProgressBar;
    ImageView spectrogramImageView, colorBarImageView;
    Spinner dataSetSpinner, colorMapSpinner, integrationSpinner, languageSpinner;
    Switch loopSwitch, compressorSwitch;

    // INITIALIZE GLOBAL VARIABLES:
    public int sampleRate=      48000;
    public int hopLength =      1024;
    public int fftSize =        2048;
    public int melFreqs =       128;
    public int frames =         128;
    public int nInteg =         5;
    public int numClasses;
    public int sizeInFloats;
    public String saveName;
    public int targetSamples = (frames+1)*hopLength; // = 132096
    public int totalSamples = sampleRate* 4; // record for 4 seconds
    public int accuracy_0=0, accuracy_1=0, accuracy_2=0;
    public float[] pred_0, pred_1, pred_2, pred_3, pred_4;
    public float[] audioSignal = new float[targetSamples];
    public float[][] spectrogram = new float[melFreqs][frames];
    private static final int STORAGE_CODE =     1;
    public static final int RECORD_AUDIO_CODE = 200;
    public boolean loopRunning = true;
    public boolean compressorOn = false;
    public String[] labels;
    public String modelFileName, dynamicRange, label_0="...", label_1="...", label_2="...";
    public String currentLanguage = "ENGLISH";
    public String currentColorMap = "MAGMA";
    public String currentDataSet = "PARTICULAR";
    public List<String> particularLabelsList, globalLabelsList;
    public String directory = Environment.getExternalStorageDirectory()+"/SceneClassifier";

    //----------------------------------- METHODS -------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        automaticCheckAudioPermission();
        automaticCheckStoragePermission();
        setupUI();
        resetUI();
        initializeLogger();
        Log.d("console", "UI: App Created!");
    } // onCreate method end

    private void setAppLocale(String code){
        Resources res = getResources();
        DisplayMetrics dm = res.getDisplayMetrics();
        Configuration conf = res.getConfiguration();
        conf.setLocale(new Locale(code.toLowerCase()));
        res.updateConfiguration(conf,dm);
    }

    private void setupUI(){
        // CALL LAYOUT OBJECTS:
        validateButton =            findViewById(R.id.validateButton);
        logButton =                 findViewById(R.id.logButton);
        predictionView_0 =          findViewById(R.id.prediction1TextView);
        predictionView_1 =          findViewById(R.id.prediction2TextView);
        predictionView_2 =          findViewById(R.id.prediction3TextView);
        bufferTextView =            findViewById(R.id.bufferTextView);
        bufferEditTextView =        findViewById(R.id.bufferEditTextView);
        xLabelTextView =            findViewById(R.id.xLabelTextView);
        dynamicRangeTextViewEdit =  findViewById(R.id.minTextView);
        dynamicRangeTextView =      findViewById(R.id.maxTextView);
        languageTextView =          findViewById(R.id.languageTextView);
        spectrogramImageView =      findViewById(R.id.spectrogramImageView);
        colorBarImageView =         findViewById(R.id.colorbarImageView);
        predictionBar1=             findViewById(R.id.progressBar1);
        predictionBar2=             findViewById(R.id.progressBar2);
        predictionBar3=             findViewById(R.id.progressBar3);
        loadingBar =                findViewById(R.id.loadingBar);
        horizontalProgressBar =     findViewById(R.id.horizontalProgressBar);
        dataSetTextView =           findViewById(R.id.datasetTextView);
        colorMapTextView =          findViewById(R.id.colorMapTextView);
        integrationTextView =       findViewById(R.id.integrationTextView);
        dataSetSpinner =            findViewById(R.id.datasetSpinner);
        colorMapSpinner =           findViewById(R.id.colorMapSpinner);
        integrationSpinner =        findViewById(R.id.integrationSpinner);
        runButton =                 findViewById(R.id.runButton);
        infoButton =                findViewById(R.id.infoButton);
        languageSpinner =           findViewById(R.id.languageSpinner);
        loopSwitch =                findViewById(R.id.loopSwitch);
        compressorSwitch =          findViewById(R.id.compressorSwitch);
        Log.d("console", "UI: Objects Loaded!");
        // LANGUAGE SPINNER:
        ArrayAdapter<CharSequence> languageAdapter = ArrayAdapter.createFromResource(
                this,R.array.language,android.R.layout.simple_spinner_item);
        languageAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        languageSpinner.setAdapter(languageAdapter);
        languageSpinner.setOnItemSelectedListener(this);
        // DATA BASE SPINNER:
        ArrayAdapter<CharSequence> dataSetAdapter = ArrayAdapter.createFromResource(
                this,R.array.dataset,android.R.layout.simple_spinner_item);
        dataSetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        dataSetSpinner.setAdapter(dataSetAdapter);
        dataSetSpinner.setOnItemSelectedListener(this);
        // INTEGRATION SPINNER:
        ArrayAdapter<CharSequence> integrationAdapter = ArrayAdapter.createFromResource(
                this,R.array.integration,android.R.layout.simple_spinner_item);
        integrationAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        integrationSpinner.setAdapter(integrationAdapter);
        integrationSpinner.setOnItemSelectedListener(this);
        // COLORMAP SPINNER:
        ArrayAdapter<CharSequence> colorMapAdapter = ArrayAdapter.createFromResource(
                this,R.array.colorMap,android.R.layout.simple_spinner_item);
        colorMapAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        colorMapSpinner.setAdapter(colorMapAdapter);
        colorMapSpinner.setOnItemSelectedListener(this);
        // BUTTONS:
        validateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {validationDataDialog();}});
        logButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {logPredictions();}});
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {executeTasks();}});
        infoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {showInfo();}});
        // SWITCH:
        loopSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                loopRunning = isChecked;}});
        compressorSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                compressorOn = isChecked;}});
    } // setupUI end

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String itemSelected = parent.getItemAtPosition(position).toString();
        // SPINNER SELECT LANGUAGE;
        if (itemSelected.equals("ENGLISH")) {currentLanguage = "ENGLISH";}
        if (itemSelected.equals("ESPAÑOL")) {currentLanguage = "ESPAÑOL";}
        // SPINNER SELECT MODEL;
        if (itemSelected.equals("PARTICULAR")) {
            currentDataSet = "PARTICULAR";}
        if (itemSelected.equals("GLOBAL")) {
            currentDataSet = "GLOBAL";}
        // SPINNER SELECT INTEGRATION;
        if (itemSelected.equals("INT 5")) {nInteg = 5;}
        if (itemSelected.equals("INT 4")) {nInteg = 4;}
        if (itemSelected.equals("INT 3")) {nInteg = 3;}
        if (itemSelected.equals("INT 2")) {nInteg = 2;}
        if (itemSelected.equals("INT 1")) {nInteg = 1;}
        // SPINNER SELECT COLORMAP:
        if (itemSelected.equals("MAGMA")) {currentColorMap = "MAGMA"; }
        if (itemSelected.equals("PLASMA")) {currentColorMap = "PLASMA"; }
        if (itemSelected.equals("VIRIDIS")) {currentColorMap = "VIRIDIS"; }
        setLanguage();
        updateLabels();
        initializePredictions();
        updatePredictions();
        updateStrings();
        showSpectrogram();
        Toast.makeText(this, "Settings Updated", Toast.LENGTH_SHORT).show();
        Log.d("console", "UI: Spinners selected: "+itemSelected);
    } // onItemSelected method end

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        Log.d("console", "Nothing Selected");
    } // onNothingSelected end

    private void setLanguage(){
        if (currentLanguage.equals("ENGLISH")) {
            setAppLocale("en");
            particularLabelsList = Arrays.asList(getResources().getStringArray(R.array.p_EN));
            globalLabelsList = Arrays.asList(getResources().getStringArray(R.array.g_EN));}
        if (currentLanguage.equals("ESPAÑOL")) {
            setAppLocale("es");
            particularLabelsList = Arrays.asList(getResources().getStringArray(R.array.p_ES));
            globalLabelsList = Arrays.asList(getResources().getStringArray(R.array.g_ES));}
    } // setLanguage end

    private void updateLabels(){
        if (currentDataSet.equals("PARTICULAR")) {
            modelFileName = "PARTICULAR.tflite";
            numClasses = particularLabelsList.size();
            labels = new String[numClasses];
            for (int i = 0; i < numClasses; i++) {labels[i] = particularLabelsList.get(i);
                Log.d("Labels", "LABEL: "+i+" "+labels[i]);
            }
        } else if (currentDataSet.equals("GLOBAL")){
            modelFileName = "GLOBAL.tflite";
            numClasses = globalLabelsList.size();
            labels = new String[numClasses];
            for (int i = 0; i < numClasses; i++) {labels[i] = globalLabelsList.get(i);
                Log.d("Labels", "LABEL: "+i+" "+labels[i]);
            }
        }
    } // updateLabels end

    private void updateStrings(){
        bufferTextView.setText(R.string.textBuffer);
        xLabelTextView.setText(R.string.textXLabel);
        dataSetTextView.setText(R.string.textDataSet);
        integrationTextView.setText(R.string.textIntegration);
        runButton.setText(R.string.textRun);
        languageTextView.setText(R.string.textLanguage);
        colorMapTextView.setText(R.string.textColorMap);
        dynamicRangeTextView.setText(getResources().getString(R.string.textDynamicRange));
    }

    private void resetUI() {
        logButton.setEnabled(false);
        validateButton.setEnabled(false);
        infoButton.setEnabled(false);
        dataSetSpinner.setEnabled(false);
        compressorSwitch.setEnabled(false);
        validateButton.setVisibility(View.INVISIBLE);
        logButton.setVisibility(View.INVISIBLE);
        infoButton.setVisibility(View.INVISIBLE);
        xLabelTextView.setVisibility(View.INVISIBLE);
        predictionView_0.setVisibility(View.INVISIBLE);
        predictionView_1.setVisibility(View.INVISIBLE);
        predictionView_2.setVisibility(View.INVISIBLE);
        predictionBar1.setVisibility(View.INVISIBLE);
        predictionBar2.setVisibility(View.INVISIBLE);
        predictionBar3.setVisibility(View.INVISIBLE);
        bufferTextView.setVisibility(View.INVISIBLE);
        bufferEditTextView.setVisibility(View.INVISIBLE);
        spectrogramImageView.setVisibility(View.INVISIBLE);
        colorBarImageView.setVisibility(View.INVISIBLE);
        loadingBar.setVisibility(View.INVISIBLE);
        dynamicRangeTextViewEdit.setVisibility(View.INVISIBLE);
        dynamicRangeTextView.setVisibility(View.INVISIBLE);
        loopSwitch.setVisibility(View.INVISIBLE);
        compressorSwitch.setVisibility(View.INVISIBLE);
        dataSetSpinner.setVisibility(View.INVISIBLE);
        dataSetTextView.setVisibility(View.INVISIBLE);
    }

    private void showUI(){
        logButton.setVisibility(View.VISIBLE);
        infoButton.setVisibility(View.VISIBLE);
        predictionView_0.setVisibility(View.VISIBLE);
        predictionView_1.setVisibility(View.VISIBLE);
        predictionView_2.setVisibility(View.VISIBLE);
        bufferTextView.setVisibility(View.VISIBLE);
        bufferEditTextView.setVisibility(View.VISIBLE);
        xLabelTextView.setVisibility(View.VISIBLE);
        spectrogramImageView.setVisibility(View.VISIBLE);
        colorBarImageView.setVisibility(View.VISIBLE);
        predictionBar1.setVisibility(View.VISIBLE);
        predictionBar2.setVisibility(View.VISIBLE);
        predictionBar3.setVisibility(View.VISIBLE);
        dynamicRangeTextViewEdit.setVisibility(View.VISIBLE);
        dynamicRangeTextView.setVisibility(View.VISIBLE);
        runButton.setTextColor(getColor(R.color.colorAccent));
        logButton.setTextColor(getColor(R.color.colorAccent));
        infoButton.setTextColor(getColor(R.color.colorAccent));
        loopSwitch.setVisibility(View.VISIBLE);
        loadingBar.setVisibility(View.INVISIBLE);
        runButton.setEnabled(true);
        logButton.setEnabled(true);
        infoButton.setEnabled(true);
        languageSpinner.setEnabled(true);
        integrationSpinner.setEnabled(true);
        validateButton.setTextColor(getColor(R.color.colorAccent));
        runButton.setText(R.string.textRun);
        runButton.setTextColor(getColor(R.color.colorAccent));
        horizontalProgressBar.setProgress(0);
    } // showUI end

    private void showDeveloperUI(){
        //dataSetSpinner.setEnabled(true);
        validateButton.setEnabled(true);
        //compressorSwitch.setEnabled(true);
        //compressorSwitch.setVisibility(View.VISIBLE);
        //dataSetSpinner.setVisibility(View.VISIBLE);
        validateButton.setVisibility(View.VISIBLE);
    } // showDeveloperUI end

    private void initializePredictions(){
        if (pred_0 == null || pred_0.length!=numClasses) {pred_0 = new float[numClasses];}
        if (pred_1 == null || pred_1.length!=numClasses) {pred_1 = new float[numClasses];}
        if (pred_2 == null || pred_2.length!=numClasses) {pred_2 = new float[numClasses];}
        if (pred_3 == null || pred_3.length!=numClasses) {pred_3 = new float[numClasses];}
        if (pred_4 == null || pred_4.length!=numClasses) {pred_4 = new float[numClasses];}
    }

    private void initializeLogger(){
        File folder = new File(Environment.getExternalStorageDirectory(),"SceneClassifier");
        if (folder.exists()) {  Log.d("console", "Directory Found");}
        else {  boolean created = folder.mkdirs();
            if (created) {    Log.d("console", "Directory Created");}
            else { Log.d("console", "Directory Not Created");}
        }
        File loggerFile = new File(directory,"Logs.csv");
        if (loggerFile.exists()) {
            Log.d("console", "Log File Found");
        } else {
            StringBuffer logLabelsCSV;
            logLabelsCSV = new StringBuffer("Timestamp");
            for (int i = 0; i < numClasses; i++) {
                logLabelsCSV.append(",");
                logLabelsCSV.append(labels[i]);
            }
            saveCSV(logLabelsCSV, "Logs.csv");
            Log.d("console", "Log File has been created");
        }
    } // initializeLogger end

    private void showInfo(){
        Intent intent = new Intent(MainActivity.this,InformationActivity.class);
        intent.putExtra("dataset", currentDataSet);
        intent.putExtra("language",currentLanguage);
        intent.putExtra("classes", numClasses+"°");
        startActivity(intent);
    }

    private void executeTasks(){
        RecordTask recordTask  = new RecordTask();
        recordTask.execute();
    }

    private void showSpectrogram() {
        double[][] spectrogramToPlot = new double[frames][melFreqs];
        // publish min and max values.
        float specMax=-100;
        float specMin=100;
        for (int x = 0; x< frames; x++ ) {
            for (int y = 0; y< melFreqs; y++){
                spectrogramToPlot[y][x] = spectrogram[x][y];
                if (spectrogram[x][y]>specMax){specMax = spectrogram[x][y];}
                if (spectrogram[x][y]<specMin){specMin = spectrogram[x][y];}}}
        dynamicRange = (80.0f-(Math.round(specMin*80.0*10.0)/10.0))+" dB";
        // apply color map.
        SpectrogramView spectrogramView = new SpectrogramView(
                getApplicationContext(),spectrogramToPlot,currentColorMap,5,10);
        spectrogramImageView.setImageBitmap(spectrogramView.getBitmap());
        if (currentColorMap.equals("MAGMA")){
            colorBarImageView.setImageResource(R.drawable.magma_cmap);}
        if (currentColorMap.equals("PLASMA")){
            colorBarImageView.setImageResource(R.drawable.plasma_cmap);}
        if (currentColorMap.equals("VIRIDIS")){
            colorBarImageView.setImageResource(R.drawable.viridis_cmap);}
        Log.d("console", "UI: spectrogram Updated");
        dynamicRangeTextViewEdit.setText(dynamicRange);
        dynamicRangeTextView.bringToFront();
        dynamicRangeTextViewEdit.bringToFront();
    } // end

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    } // loadModelFile method end

    private void savePredictions(float[][] outputTensorArray){
        if (pred_0 ==null || pred_0.length!=numClasses || nInteg <2) {
            pred_0 = new float[numClasses];}
        if (pred_1 ==null || pred_1.length!=numClasses || nInteg <3) {
            pred_1 = new float[numClasses];}
        if (pred_2 ==null || pred_2.length!=numClasses || nInteg <4) {
            pred_2 = new float[numClasses];}
        if (pred_3 ==null || pred_3.length!=numClasses || nInteg <5) {
            pred_3 = new float[numClasses];}
        pred_4 = new float[numClasses];
        pred_4 = pred_3.clone();
        pred_3 = pred_2.clone();
        pred_2 = pred_1.clone();
        pred_1 = pred_0.clone();
        pred_0 = outputTensorArray[0].clone();
        Log.d("NNA:", "Predictions Saved");
    }

    private void updatePredictions(){
        float[] meanPredictions = new float[numClasses];
        for (int i = 0 ; i<numClasses; i++ ){
            meanPredictions[i] = (pred_0[i]+ pred_1[i]+ pred_2[i]+ pred_3[i]+ pred_4[i])/ nInteg;}
        float[] bestPredictions = meanPredictions.clone();
        Arrays.sort(bestPredictions); // sort ascending
        int[] topLabelIndexes = new int[3];
        for (int i=0; i<numClasses;i++){
            if (meanPredictions[i]==bestPredictions[numClasses-1]){topLabelIndexes[0]= i;}
            if (meanPredictions[i]==bestPredictions[numClasses-2]){topLabelIndexes[1]= i;}
            if (meanPredictions[i]==bestPredictions[numClasses-3]){topLabelIndexes[2]= i;}
        }
        label_0 = labels[topLabelIndexes[0]];
        label_1 = labels[topLabelIndexes[1]];
        label_2 = labels[topLabelIndexes[2]];
        accuracy_0 = (int)(bestPredictions[numClasses-1]*100);
        accuracy_1 = (int)(bestPredictions[numClasses-2]*100);
        accuracy_2 = (int)(bestPredictions[numClasses-3]*100);
        // PASS RESULTS TO UI OBJECTS
        String predictionText_0 = label_0 + " (" + accuracy_0 + "%)";
        String predictionText_1 = label_1 + " (" + accuracy_1 + "%)";
        String predictionText_2 = label_2 + " (" + accuracy_2 + "%)";
        String bufferText      = sizeInFloats +" "+ getResources().getString(R.string.textSamples);
        predictionView_0.setText(predictionText_0);
        predictionView_1.setText(predictionText_1);
        predictionView_2.setText(predictionText_2);
        bufferEditTextView.setText(bufferText);
        predictionBar1.setProgress(accuracy_0);
        predictionBar2.setProgress(accuracy_1);
        predictionBar3.setProgress(accuracy_2);
        Log.d("console", "UI: Objects Updated");
    } //updatePredictions end

    private void logPredictions(){
        StringBuffer predictionLogCSV;
        predictionLogCSV = new StringBuffer("\n");
        predictionLogCSV.append(getCurrentTime());
        for (int i = 0; i< numClasses; i++){
            predictionLogCSV.append(",");
            predictionLogCSV.append(pred_0[i]);}
        saveCSV(predictionLogCSV,"Logs.csv");
        logButton.setEnabled(false);
        logButton.setTextColor(getColor(R.color.colorGray));
        Log.d("console", "Predictions Logged to File");
    } // logPredictions end

    private String getCurrentTime(){
        return new SimpleDateFormat("HHmmss").format(Calendar.getInstance().getTime());
    } // getCurrentTime end

    private void requestAudioPermission(){
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this,Manifest.permission.RECORD_AUDIO)) {
            new AlertDialog.Builder(this)
                    .setTitle("Permission needed")
                    .setMessage("Permission required in order to predict acoustic scenes")
                    .setPositiveButton("ok", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            ActivityCompat.requestPermissions(MainActivity.this,new String[]{
                                            Manifest.permission.RECORD_AUDIO},RECORD_AUDIO_CODE);}})
                    .setNegativeButton("cancel", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.dismiss();}}).create().show();} 
                        else {ActivityCompat.requestPermissions(this,new String[] {
                                Manifest.permission.RECORD_AUDIO}, RECORD_AUDIO_CODE);} 
    } // requestPermission end

    private void automaticCheckAudioPermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(MainActivity.this, "Microphone Access: OK!",
                    Toast.LENGTH_LONG).show(); }
        else {requestAudioPermission();}
        Log.d("console", "UI: Recording Permissions Checked");
    } // automaticCheckPermission end

    private void automaticCheckStoragePermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(MainActivity.this, "Storage Access: OK!",
                    Toast.LENGTH_SHORT).show(); }
        else {requestStoragePermission();}
        Log.d("console", "UI: Recording Permissions Checked");
    } // automaticCheckPermission end

    private void requestStoragePermission(){
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this,Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            new AlertDialog.Builder(this)
                    .setTitle("Permission needed")
                    .setMessage("Permission required in order to save data")
                    .setPositiveButton("ok", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            ActivityCompat.requestPermissions(MainActivity.this,new String[]{
                                    Manifest.permission.WRITE_EXTERNAL_STORAGE},STORAGE_CODE);}})
                    .setNegativeButton("cancel", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.dismiss();}}).create().show();}
        else {ActivityCompat.requestPermissions(this,new String[] {
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, STORAGE_CODE);}
    } // requestStoragePermission end

    private boolean isExternalStorageWritable(){
        if(Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())){
            return true;}
        else { Log.d("console", "External Storage is not Writable"); return false; }
    } // isExternalStorageWritable end

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == RECORD_AUDIO_CODE)  {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission GRANTED", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission DENIED", Toast.LENGTH_SHORT).show();}}
    } // onRequestPermissionsResult end

    private void validationDataDialog(){
        saveName = "NewFile";
        AlertDialog.Builder alertBuilder = new AlertDialog.Builder(MainActivity.this);
        View dialogView = getLayoutInflater().inflate(R.layout.input_dialog,null);
        final EditText userInputText = dialogView.findViewById(R.id.userInputText);
        alertBuilder.setView(dialogView);
        alertBuilder.setCancelable(true)
                .setTitle("Validation Data Name")
                .setPositiveButton("OK",new DialogInterface.OnClickListener(){
                    @Override
                    public void onClick(DialogInterface dialog,int which){
                        if(!userInputText.getText().toString().isEmpty()) {
                            dialog.dismiss();
                            saveName = "_" + userInputText.getText().toString();
                            saveValidationData();
                        }}});
        Dialog dialog = alertBuilder.create();
        dialog.show();
    }

    private void saveValidationData(){
        // INITIALIZE OBJECTS:
        StringBuffer signalDataCSV;
        StringBuffer spectrogramDataCSV;
        StringBuffer predictionsDataCSV;
        // INITIALIZE LABELS:
        signalDataCSV = new StringBuffer("index" + "," + "Amplitude");
        spectrogramDataCSV = new StringBuffer();
        predictionsDataCSV = new StringBuffer("index" + "," + "class" + "," + "output");
        // BUILD SIGNAL STRING BUFFER:
        for (int i = 0; i< targetSamples; i++){
            signalDataCSV.append("\n");
            signalDataCSV.append(i);
            signalDataCSV.append("," );
            signalDataCSV.append(audioSignal[i]);}
        // BUILD SPECTROGRAM STRING BUFFER:
        for (int i = 0; i< melFreqs; i++){
            for (int j = 0; j< frames; j++){
                spectrogramDataCSV.append(spectrogram[i][j]);
                spectrogramDataCSV.append(",");}
            spectrogramDataCSV.append("\n");}
        // BUILD PREDICTIONS STRING BUFFER:
        for (int i = 0; i< numClasses;i++){
            predictionsDataCSV.append("\n");
            predictionsDataCSV.append(i);
            predictionsDataCSV.append(",");
            predictionsDataCSV.append(labels[i]);
            predictionsDataCSV.append(",");
            predictionsDataCSV.append(pred_0[i]);}
        // SAVE CSV FILES:
        saveCSV(signalDataCSV,getCurrentTime()+saveName+"_signal.csv");
        saveCSV(spectrogramDataCSV,getCurrentTime()+saveName+"_spectrogram.csv");
        saveCSV(predictionsDataCSV,getCurrentTime()+saveName+"_predictions.csv");
        validateButton.setTextColor(getColor(R.color.colorGray));
        Log.d("console", "CSV Files Saved to Memory");
    } // saveValidationData end

    private void saveCSV(StringBuffer stringBufferData, String fileName ){
        if (isExternalStorageWritable()) {
            File currentFile = new File(directory, fileName);
            String data = stringBufferData.toString(); // convert StringBuffer to String
            try {
                FileOutputStream fos = new FileOutputStream(currentFile,true);
                fos.write(data.getBytes());
                fos.close();
                Toast.makeText(this,fileName+" "+
                        getResources().getString(R.string.textSaved),Toast.LENGTH_SHORT).show();
            } catch (IOException e) { e.printStackTrace();
            }} else {Toast.makeText(this,fileName+" "+
                        getResources().getString(R.string.textNot_saved),Toast.LENGTH_SHORT).show();
        }
    } // saveCSV end

    private class RecordTask extends AsyncTask<Void, Integer, float[]> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            runButton.setText(R.string.textREC);
            runButton.setTextColor(getColor(R.color.colorlightRed));
            runButton.setEnabled(false);
        }
        @Override
        protected float[] doInBackground(Void... voids) {
            int source = MediaRecorder.AudioSource.UNPROCESSED;
            int channel = AudioFormat.CHANNEL_IN_MONO;
            int format = AudioFormat.ENCODING_PCM_FLOAT;
            int readMode = AudioRecord.READ_BLOCKING;
            boolean isRecording;
            int minBuffer = AudioRecord.getMinBufferSize(sampleRate, channel, format);
            sizeInFloats = minBuffer / 4; // FLOAT32 --> 32 bits --> 4 bytes
            float[] audioMiniBuffer = new float[sizeInFloats]; // small temp buffer
            float[] recordedAudio   = new float[totalSamples]; // 3 sec of samples
            float[] targetAudio     = new float[targetSamples]; // target signal
            AudioRecord recorder = new AudioRecord(source, sampleRate, channel, format, minBuffer);
            if (recorder.getState() == 0) {
                Log.d("console", " ERROR : AudioRecord uninitialized");
            } else if (recorder.getState() == 1) {
                Log.d("console", "REC: Recording!...");
                recorder.startRecording();
                isRecording = true;
                int offset = 0;
                int readCounter;
                int sampleIndex = 0;
                int totalRead = 0;
                while (isRecording) {
                    readCounter = recorder.read(audioMiniBuffer, offset, sizeInFloats, readMode);
                    totalRead += readCounter;
                    publishProgress(totalRead);
                    //Log.d("rec_while", "current sample: "+sampleIndex);
                    if (AudioRecord.ERROR_INVALID_OPERATION != readCounter) { // error check.
                        for (int i = 0; i < sizeInFloats; i++) {
                            if (sampleIndex < totalSamples) { // need more samples!
                                recordedAudio[sampleIndex] = audioMiniBuffer[i];}
                            if (sampleIndex >= totalSamples) { // enough samples!
                                //Log.d("rec_while", "REC: Enough Samples!");
                                if (recorder.getState() == 1) {
                                    recorder.stop();
                                    recorder.release();
                                    recorder = null;
                                    isRecording = false;
                                    Log.d("console", "REC: Recording stopped!");
                                    break;}}
                            sampleIndex += 1;}}}
                int sourcePos = totalSamples - targetSamples; // use last recorded samples.
                System.arraycopy(recordedAudio, sourcePos, targetAudio, 0, targetSamples);}
            // PRE-PROCESS:
            targetAudio = normalize(targetAudio);
//            if (compressorOn){
//                targetAudio = compressor(targetAudio);
//                targetAudio = normalize(targetAudio);}
            // DESCRIPTORS:
            float totalDuration = ((float)(targetAudio.length))/sampleRate;
            Log.d("console", "Signal Min:"+minimum(targetAudio));
            Log.d("console", "Signal Max:"+maximum(targetAudio));
            Log.d("console", "Signal Average:"+average(targetAudio));
            Log.d("console", "REC: total duration: "+totalDuration+" s");
            return targetAudio;
        }
        private float maximum(float[] signal){
            float signalMax = signal[0];
            for (float v : signal) { if (v > signalMax) { signalMax = v; }}
            return signalMax;
        }
        private float minimum(float[] signal){
            float signalMin = signal[0];
            for (float v : signal) { if (v < signalMin) { signalMin = v; }}
            return signalMin;
        }
        private float average(float[] signal){
            float signalAverage=0;
            for (float v : signal) { signalAverage+=v;}
            signalAverage/=signal.length;
            return signalAverage;
        }
        private float peak(float[] signal){
            return Math.max(Math.abs(maximum(signal)), Math.abs(minimum(signal)));
        }
        private float rms(float[] signal){
            float rms = 0.0f;
            for (float v : signal) {rms += (v * v);}
            return (float) Math.sqrt(rms/signal.length);
        }
        private float[] normalize(float[] signal) {
            float[] normalizedSignal = new float[signal.length];
            float peak = Math.max(Math.abs(maximum(signal)),Math.abs(minimum(signal)));
            for (int i = 0; i < signal.length; i++) {normalizedSignal[i] = signal[i] / peak;}
            return normalizedSignal;
        }
        private float[] compressor(float[] signal) {
            float threshold = 1-((peak(signal)-rms(signal))); // variable threshold.
            float gainReduction = 1/2;
            float gainMakeUp = 1.0f;
            float[] output = new float[signal.length];
            System.arraycopy(signal, 0, output, 0, signal.length);
            for (int i = 0; i < signal.length; i++) {
                if (signal[i] > threshold) {
                    output[i] = +threshold + (Math.abs(threshold - signal[i]))*gainReduction;}
                if (signal[i] < -threshold) {
                    output[i] = -threshold -(Math.abs(-threshold - signal[i]))*gainReduction;}
                output[i] = output[i] * gainMakeUp;
                if (output[i] > +1.0f) {output[i] = +1.0f;}
                if (output[i] < -1.0f) {output[i] = -1.0f;}
            }
            return output;
        }
        @Override
        protected void onProgressUpdate(Integer... samplesRead) {
            super.onProgressUpdate(samplesRead);
            int progress = (int)(((float)samplesRead[0]/totalSamples)*100);
            horizontalProgressBar.setProgress(progress);
        }
        @Override
        protected void onPostExecute(float[] outputAudio) {
            super.onPostExecute(outputAudio);
            System.arraycopy(outputAudio, 0, audioSignal, 0, audioSignal.length);
            ProcessTask processTask  = new ProcessTask();
            processTask.execute(outputAudio);
        }
    } // RecordTask end

    private class ProcessTask extends AsyncTask<float[], Integer, float[][]> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            Log.d("console", "PROCESS: started");
            loadingBar.setVisibility(View.VISIBLE);
            runButton.setTextColor(getColor(R.color.colorlightBlue));
            runButton.setText(R.string.textProcess);
        }
        @Override
        protected float[][] doInBackground(float[]... signals) {
            publishProgress(0);
            float[] signal = signals[0];
            float[][] melSpec = new MelSpectrogram(
                    signal, sampleRate, melFreqs, frames, fftSize,hopLength).getSpectrogram();
            publishProgress(100);
            return melSpec;
        }
        @Override
        protected void onProgressUpdate(Integer... progress) {
            super.onProgressUpdate(progress);
            horizontalProgressBar.setProgress(progress[0]);
        }
        @Override
        protected void onPostExecute(float[][] melSpec) {
            super.onPostExecute(melSpec);
            System.arraycopy(melSpec,0,spectrogram,0,melSpec.length);
            showSpectrogram();
            InferenceTask inferenceTask  = new InferenceTask();
            inferenceTask.execute(melSpec);}
    } // ProcessTask end

    private class InferenceTask extends AsyncTask<float[][], Integer, float[][]> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            languageSpinner.setEnabled(false);
            integrationSpinner.setEnabled(false);
            runButton.setTextColor(getColor(R.color.colorlightGreen));
            runButton.setText(R.string.textInference);
        }
        @Override
        protected float[][] doInBackground(float[][]... specs) {
            publishProgress(0);
            float[][] spec = specs[0];
            float[][][][] inputTensor = new float[1][melFreqs][frames][1];
            for (int i = 0; i< frames; i++){
                for (int j = 0; j< melFreqs; j++) {
                    inputTensor[0][i][j][0] = spec[i][j];}}
            float[][] outputTensor = new float[1][numClasses];
            try {
                publishProgress(10);
                Interpreter tensorFlowLite = new Interpreter(loadModelFile());
                publishProgress(50);
                tensorFlowLite.run(inputTensor,outputTensor);
                tensorFlowLite.close();
                Log.d("NNA:", "Inference executed | Classes: "+ numClasses);
                publishProgress(100);
            } catch (Exception ex){ex.printStackTrace();}
            return outputTensor;
        }
        @Override
        protected void onProgressUpdate(Integer... progress) {
            super.onProgressUpdate(progress);
            horizontalProgressBar.setProgress(progress[0]);
        }
        @Override
        protected void onPostExecute(float[][] outputTensor) {
            super.onPostExecute(outputTensor);
            savePredictions(outputTensor);
            updatePredictions();
            showUI();
            showDeveloperUI();
            if (loopRunning) {executeTasks();}}
    } // ProcessTask end
} // MainActivity class end;

