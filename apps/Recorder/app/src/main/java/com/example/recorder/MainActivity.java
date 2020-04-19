package com.example.recorder;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    Button recButton, saveButton, stopButton;
    Spinner sampleRateSpinner, bitDepthSpinner;
    TextView timeTextView;
    ProgressBar progressBar,levelMeter;
    EditText editText;

    private static final int STORAGE_CODE = 1;
    public static final int RECORD_AUDIO_CODE = 200;
    public String directory = Environment.getExternalStorageDirectory() + "/Recorder";
    public int sampleRate;
    public int bitDepth;
    public boolean compressorOn=false;
    public boolean userStop=false;
    public int totalRead;
    public int maxSamples;
    public float[] audioSignal;
    public float signalMeter;
    @Override

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        automaticCheckAudioPermission();
        automaticCheckStoragePermission();
        setupUI();
    } // onCreate end

    private void setupUI() {
        recButton = findViewById(R.id.recButton);
        saveButton = findViewById(R.id.saveButton);
        stopButton = findViewById(R.id.stopButton);
        sampleRateSpinner = findViewById(R.id.sampleRateSpinner);
        bitDepthSpinner = findViewById(R.id.bitDepthSpinner);
        progressBar = findViewById(R.id.progressBar);
        timeTextView = findViewById(R.id.timeTextView);
        levelMeter = findViewById(R.id.levelMeter);
        editText = findViewById(R.id.editText);
        // BUTTONS:
        recButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                RecordTask recordTask  = new RecordTask();
                recordTask.execute();}
        });
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) { getSaveName();
            }});
        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v){
                userStop=true;
            }});
        // SAMPLE RATE SPINNER:
        ArrayAdapter<CharSequence> sampleRateAdapter = ArrayAdapter.createFromResource(
                this, R.array.sampleRateOptions, android.R.layout.simple_spinner_item);
        sampleRateAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sampleRateSpinner.setAdapter(sampleRateAdapter);
        sampleRateSpinner.setOnItemSelectedListener(this);
        // BIT DEPTH SPINNER:
        ArrayAdapter<CharSequence> bitDepthAdapter = ArrayAdapter.createFromResource(
                this, R.array.bitDepthOptions, android.R.layout.simple_spinner_item);
        bitDepthAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        bitDepthSpinner.setAdapter(bitDepthAdapter);
        bitDepthSpinner.setOnItemSelectedListener(this);
        // BUTTON APPEARANCE:
        stopButton.setEnabled(false);
        saveButton.setEnabled(false);
        editText.setEnabled(true);
        editText.setVisibility(View.INVISIBLE);
        saveButton.setVisibility(View.INVISIBLE);
        recButton.setTextColor(getColor(R.color.colorAccent));
        initializeAppFolder();
    } // setupUI end

    @Override
     public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String itemSelected = parent.getItemAtPosition(position).toString();
        int inputNumber = Integer.valueOf(itemSelected);
        if (inputNumber > 100){sampleRate = inputNumber;}
        if (inputNumber < 100){bitDepth = inputNumber;}
    } // onItemSelected end

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    } // onNothingSelected end;

    public void initializeAppFolder(){
        File folder = new File(Environment.getExternalStorageDirectory(),"Recorder");
        if (folder.exists()) {  Log.d("console", "Directory Found");}
        else {  boolean created = folder.mkdirs();
            if (created) {    Log.d("console", "Directory Created");}
            else { Log.d("console", "Directory Not Created");}
        }
    } // initializeAppFolder end

    @SuppressLint("SimpleDateFormat")
    private String getCurrentTime(){
        return new SimpleDateFormat("HHmmss").format(Calendar.getInstance().getTime());
    } // getCurrentTime end

    private void getSaveName(){
        String saveName = getCurrentTime()+"_"+editText.getText().toString()+".wav";
        saveWavData(saveName);
    } // getSaveName end

    private void saveWavData(String saveName){
        int frames = audioSignal.length;
        int channels = 1;
        double[] sampleBuffer = new double[frames];
        for (int i=0;i<frames;i++){ sampleBuffer[i] = audioSignal[i]; }

        File newFile = new File(directory,saveName);
        try {
            WavFile wavFile = WavFile.newWavFile(newFile,channels,frames, bitDepth, sampleRate);
            wavFile.writeFrames(sampleBuffer,frames);
            wavFile.close();
        } catch (IOException | WavFileException e) { e.printStackTrace();}
        Toast.makeText(this, saveName + " saved", Toast.LENGTH_SHORT).show();
        editText.setEnabled(false);
        editText.setVisibility(View.INVISIBLE);
    } // saveWavData end

    private void requestAudioPermission(){
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this, Manifest.permission.RECORD_AUDIO)) {
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

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == RECORD_AUDIO_CODE)  {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission GRANTED", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission DENIED", Toast.LENGTH_SHORT).show();}}
    } // onRequestPermissionsResult end


private class RecordTask extends AsyncTask<Void, Integer, float[]> {

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        recButton.setTextColor(getColor(R.color.colorlightRed));
        sampleRateSpinner.setEnabled(false);
        stopButton.setEnabled(true);
        saveButton.setEnabled(false);
        bitDepthSpinner.setEnabled(false);
        recButton.setTextColor(Color.BLACK);
        saveButton.setTextColor(Color.GRAY);
        stopButton.setTextColor(Color.RED);
        recButton.setText("RECORDING");
        editText.setVisibility(View.INVISIBLE);
        editText.setEnabled(false);
    }
    @Override
    protected float[] doInBackground(Void... voids) {
        maxSamples = 60*10*sampleRate;
        int source = MediaRecorder.AudioSource.UNPROCESSED;
        int channel = AudioFormat.CHANNEL_IN_MONO;
        int format = AudioFormat.ENCODING_PCM_FLOAT;
        int readMode = AudioRecord.READ_BLOCKING;
        boolean isRecording;
        int minBuffer = AudioRecord.getMinBufferSize(sampleRate, channel, format);
        int sizeInFloats = minBuffer / 4; // FLOAT32 --> 32 bits --> 4 bytes
        float[] audioMiniBuffer = new float[sizeInFloats]; // small temp buffer
        float[] recordedAudio   = new float[maxSamples]; // 3 sec of samples
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
            totalRead = 0;
            while (isRecording) {
                if (sampleIndex % 2 ==0){
                signalMeter= (float)(10*Math.log(maximum(audioMiniBuffer)/0.0002));}
                readCounter = recorder.read(audioMiniBuffer, offset, sizeInFloats, readMode);
                totalRead += readCounter;
                //Log.d("rec_while", "current sample: "+sampleIndex);
                if (AudioRecord.ERROR_INVALID_OPERATION != readCounter) { // error check.
                    for (int i = 0; i < sizeInFloats; i++) {
                        if (sampleIndex < maxSamples) { // need more samples!
                            recordedAudio[sampleIndex] = audioMiniBuffer[i];

                        }
                        if (sampleIndex >= maxSamples || userStop) { // enough samples!
                            //Log.d("rec_while", "REC: Enough Samples!");
                            if (recorder.getState() == 1) {
                                recorder.stop();
                                recorder.release();
                                recorder = null;
                                isRecording = false;
                                Log.d("console", "REC: Recording stopped!");
                                break;}}
                        sampleIndex += 1;
                    }
                }
                publishProgress(totalRead);            }            }
        audioSignal = new float[totalRead];
        System.arraycopy(recordedAudio, 0, audioSignal, 0, totalRead);
        // PRE-PROCESS:
        audioSignal = normalize(audioSignal);
        if (compressorOn){
            audioSignal = compressor(audioSignal);
            audioSignal = normalize(audioSignal);}
        // DESCRIPTORS:
        float totalDuration = ((float)(audioSignal.length))/sampleRate;
        Log.d("console", "Signal Min:"+minimum(audioSignal));
        Log.d("console", "Signal Max:"+maximum(audioSignal));
        Log.d("console", "Signal Average:"+average(audioSignal));
        Log.d("console", "REC: total duration: "+totalDuration+" s");
        return audioSignal;
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
        float threshold = 1-((peak(signal)-rms(signal)));
        float gainReduction = 1/4;
        float[] output = new float[signal.length];
        System.arraycopy(signal, 0, output, 0, signal.length);
        for (int i = 0; i < signal.length; i++) {
            if (signal[i] > threshold) {
                output[i] = +threshold + (Math.abs(threshold - signal[i]))*gainReduction;}
            if (signal[i] < -threshold) {
                output[i] = -threshold -(Math.abs(-threshold - signal[i]))*gainReduction;}
            if (output[i] > +1.0f) {output[i] = +1.0f;}
            if (output[i] < -1.0f) {output[i] = -1.0f;}
        }
        return output;
    }
    @Override
    protected void onProgressUpdate(Integer... progressUpdate) {
        super.onProgressUpdate(progressUpdate);
        progressBar.setProgress((int)(((float)progressUpdate[0]/maxSamples)*100));
        int led = (int)(signalMeter*1.3);
        if (led>99){led=99;}
        levelMeter.setProgress(led);
        String totalTime = "Recording: "+ (totalRead/sampleRate) + " s";
        timeTextView.setText(totalTime);
    }
    @Override
    protected void onPostExecute(float[] outputAudio) {
        super.onPostExecute(outputAudio);
        audioSignal = outputAudio;
        recButton.setText("START");
        saveButton.setEnabled(true);
        sampleRateSpinner.setEnabled(true);
        stopButton.setEnabled(false);
        userStop=false;
        editText.setEnabled(true);
        bitDepthSpinner.setEnabled(true);
        editText.setVisibility(View.VISIBLE);
        saveButton.setTextColor(getColor(R.color.colorAccent));
        String totalTime = "Recorded: "+ (totalRead/sampleRate) + " s";
        recButton.setTextColor(Color.BLACK);
        stopButton.setTextColor(Color.GRAY);
        timeTextView.setText(totalTime);
        progressBar.setProgress(0);
        levelMeter.setProgress(0);
        saveButton.setVisibility(View.VISIBLE);
    }
} // RecordTask end

}// class end
