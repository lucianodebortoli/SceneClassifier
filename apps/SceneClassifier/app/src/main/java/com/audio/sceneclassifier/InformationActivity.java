package com.audio.sceneclassifier;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;
import java.util.Arrays;
import java.util.List;

public class InformationActivity extends AppCompatActivity {

    TextView datasetTitleTextView,diversityTextView;
    ListView datasetListView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_information);
        datasetTitleTextView = findViewById(R.id.datasetTitleTextView);
        diversityTextView =    findViewById(R.id.diversityTextView);
        datasetListView =       findViewById(R.id.datasetListView);
        Intent incomingIntent = getIntent();
        String currentDataset = incomingIntent.getStringExtra("dataset");
        String currentLanguage = incomingIntent.getStringExtra("language");
        String numClasses = incomingIntent.getStringExtra("classes");
        String title = currentDataset+" - "+getResources().getString(R.string.textLabels);
        datasetTitleTextView.setText(title);
        diversityTextView.setText(numClasses);
        if (currentDataset.equals("GLOBAL") && currentLanguage.equals("ENGLISH")) {
            List<String> labels = Arrays.asList(getResources().getStringArray(R.array.g_EN));
            updateLists(labels);
        } else if (currentDataset.equals("GLOBAL") && currentLanguage.equals("ESPAÑOL")) {
            List<String> labels = Arrays.asList(getResources().getStringArray(R.array.g_ES));
            updateLists(labels);
        } else if (currentDataset.equals("PARTICULAR") && currentLanguage.equals("ENGLISH")) {
            List<String> labels = Arrays.asList(getResources().getStringArray(R.array.p_EN));
            updateLists(labels);
        } else if (currentDataset.equals("PARTICULAR") && currentLanguage.equals("ESPAÑOL")) {
            List<String> labels = Arrays.asList(getResources().getStringArray(R.array.p_ES));
            updateLists(labels);}
    } // onCreate end

    private void updateLists(List<String> labels){
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_list_item_1, labels);
        datasetListView.setAdapter(adapter);
    }

} // InformationActivity end
