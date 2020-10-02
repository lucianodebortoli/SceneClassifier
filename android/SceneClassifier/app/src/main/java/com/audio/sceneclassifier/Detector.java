package com.audio.sceneclassifier;

import android.util.Log;

import java.util.List;

public class Detector {

    // Global variables:
    public float[][] detections;
    public float[][] smoothDetections;
    private int memorySize;
    private int labelSize;


    // Constructor for Detector
    Detector(List<String> labels, int memorySize){
        setBufferSize(labels.size(), memorySize);
        Log.d("detector","labelSize "+ labelSize);
    } // Detector constructor end

    Detector(String[] labels, int memorySize){
        setBufferSize(labels.length, memorySize);
        Log.d("detector","labelSize "+ labelSize);
    } // Detector constructor end


    public synchronized float meanSmoothing(int label) {
        float sum = 0;
        for (int frame=0; frame<memorySize; frame++)
            sum += detections[label][frame];
        return sum / memorySize;
    } // mean end


    public synchronized float weigthedSmoothing(int label) {
        float sum = 0;
        int weights = 0;
        for (int frame=0; frame<memorySize; frame++){
            int weight = memorySize - frame;
            sum += detections[label][frame] * weight;
            weights += weight; }
        return sum / weights;
    } // weigthedSmoothing end


    public void add(float[] newProbabilities){
        // Shift old detections by one and add new one.
        for (int label=0; label<labelSize; label++){
            // Shift arrays:
            System.arraycopy(detections[label], 0, detections[label], 1, memorySize-1);
            System.arraycopy(smoothDetections[label], 0, smoothDetections[label], 1, memorySize-1);
            // Copy new score:
            detections[label][0] = newProbabilities[label];
            // Compute new smooth value:
            smoothDetections[label][0] = weigthedSmoothing(label);
        }
    } // add end


    public synchronized void setBufferSize(int labelsLength, int memoryLength){
        labelSize = labelsLength;
        memorySize = memoryLength;
        this.detections = new float[labelSize][memorySize];
        this.smoothDetections = new float[labelSize][memorySize];
    } // setSize end

} // Detector class end
