package com.audio.sceneclassifier;

class MelSpectrogram {

    //private melSpectrogram
    private int sampleRate;
    private int nMels;
    private int frames;
    private int fftSize;
    private int hopLength;
    private float[][] melSpectrogram;

    MelSpectrogram(float[] signal, int sampleRate, int nMels, int frames, int fftSize, int hopLength){
        this.sampleRate = sampleRate;
        this.nMels = nMels;
        this.frames = frames;
        this.fftSize = fftSize;
        this.hopLength=hopLength;
        float[][] linSpec = linearSpectrogram(signal);
        float[][] melFilters = melFilterBank();
        float[][] filteredSpectrogram = dot(melFilters, linSpec);
        float[][] melSpectrogram_dB = power_to_db(filteredSpectrogram);
        float[][] melSpectrogram_dB_flip = flipUD(melSpectrogram_dB);
        melSpectrogram = normalize(melSpectrogram_dB_flip);
    }

    float[][] getSpectrogram(){
        return melSpectrogram;
    }

    private static float[] linSpace(float min, float max, int points) {
        float[] range = new float[points];
        for (int i = 0; i < points; i++){
            range[i] = min + i * (max - min) / (points-1);}
        return range;
    }

    private static float[][] dot(float[][] x, float[][] y) {
        float[][] z = new float[x.length][y[0].length];
        if(x[0].length==y.length) {
            float sum;
            for (int n = 0; n < x.length; n++) {
                for (int m = 0; m < y[0].length; m++) {
                    sum = 0;
                    for (int i = 0; i < y.length; i++) {
                        sum += x[n][i] * y[i][m];}
                    z[n][m] = sum;}}}
        return z;
    }

    private float[] hanningWindow(int size){
        float[] window = new float[size];
        int m = size / 2;
        for (int n = -m; n < m; n++)
            window[m + n] = 0.5f + 0.5f * (float)(Math.cos(n * Math.PI / (m + 1)));
        return window;
    }

    private float[] fft(float[] signal){
        // Initialize Spectrum
        float[] spectrum = new float[fftSize / 2 + 1];
        int[] reverse; // copy samples to real/imag in bit-reversed order
        // build reverse table:
        reverse = new int[fftSize];
        reverse[0] = 0;
        for (int limit = 1, bit = fftSize / 2; limit < fftSize; limit <<= 1, bit >>= 1)
            for (int i = 0; i < limit; i++)
                reverse[i + limit] = reverse[i] + bit;
        //get real and imaginary arrays:
        float[] real = new float[fftSize];
        float[] imag = new float[fftSize];
        for (int i = 0; i < fftSize; ++i){
            real[i] = signal[reverse[i]];
            imag[i] = 0.0f;}
        // Compute FFT
        for (int halfSize = 1; halfSize < real.length; halfSize *= 2){
            float k = -(float)Math.PI/halfSize;
            float phaseShiftStepR = (float)Math.cos(k);
            float phaseShiftStepI = (float)Math.sin(k);
            // current phase shift
            float phaseShiftR = 1.0f;
            float phaseShiftI = 0.0f;
            for (int fftStep = 0; fftStep < halfSize; fftStep++) {
                for (int i = fftStep; i < real.length; i += 2 * halfSize){
                    int off = i + halfSize;
                    float tr = (phaseShiftR * real[off]) - (phaseShiftI * imag[off]);
                    float ti = (phaseShiftR * imag[off]) + (phaseShiftI * real[off]);
                    real[off] = real[i] - tr;
                    imag[off] = imag[i] - ti;
                    real[i] += tr;
                    imag[i] += ti;
                }
                float tmpR = phaseShiftR;
                phaseShiftR = (tmpR * phaseShiftStepR) - (phaseShiftI * phaseShiftStepI);
                phaseShiftI = (tmpR * phaseShiftStepI) + (phaseShiftI * phaseShiftStepR);}}
        // fill the spectrum buffer with amplitudes
        for (int i = 0; i < spectrum.length; i++){
            spectrum[i] = (float) Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);}
        return spectrum;
    }

    private float[][] linearSpectrogram(float[] signal){
        float[] window = hanningWindow(fftSize); // Licensed under the Apache License
        // COMPUTE FFTs
        int fftFreqs = 	1+fftSize/2;
        float[] fftBuffer = new float[fftSize];
        float[][] linSpectrogram = new float[fftFreqs][frames];
        int startSample = 0; // initialize sample counter
        for (int column=0;column<frames;column++) {
            for (int i=0; i<fftSize; i++) {fftBuffer[i] = signal[startSample+i] * window[i];}
            float[] fftOutput = fft(fftBuffer);  // compute FFT
            for (int k=0;k<fftFreqs;k++) { linSpectrogram[k][column] = fftOutput[k];	}
            startSample += hopLength; // increment hop
        }
        float[][] linearPowerSpectrogram = new float[fftFreqs][frames];
        for (int j=0 ; j < frames ; j++ ) {
            for (int i=0 ; i < fftFreqs ; i++ ) {
                linearPowerSpectrogram[i][j] = (float) Math.pow(Math.abs(linSpectrogram[i][j]),2);}}
        return linearPowerSpectrogram;
    }

    private float[][] melFilterBank(){
        int fMax = 20000;
        int nFreqs = 1 + fftSize / 2;
        // Fill in the linear scale *(below 1KHz);
        float max_mel = (float)
                ((1000.0)/(200.0f / 3.0f) + Math.log(fMax/1000.0f)/(Math.log(6.4f)/27.0f));
        float min_mel = 0.0f;
        float[] mels = linSpace(min_mel, max_mel, nMels+2);
        float[] freqs = new float[nMels+2];
        for (int i=0; i< nMels+2; i++) { freqs[i] = (200.0f/3.0f) * mels[i];}
        // Fill the nonlinear scale *(above 1KHz) ;
        for (int i=0; i<nMels+2; i++) {
            if (mels[i] >= (1000) / (200.0 / 3)) {
                freqs[i] =(float)
                        (1000.0*Math.exp((Math.log(6.4)/27.0)*(mels[i]-(1000.0)/(200.0/3.0))));}}
        // Triangular Filter:  Center freqs of each FFT bin;
        float[] fftFreqs = linSpace(0, ((float)sampleRate) / 2, nFreqs);
        float[][] ramps = new float[nMels+2][nFreqs];
        for (int i=0; i< nMels+2; i++) {
            for (int j=0; j < nFreqs; j++) {
                ramps[i][j]= freqs[i] - fftFreqs[j];}}
        // lower and upper slopes for all bins.;
        float[] fdiff = new float[nMels+1];
        for (int i=0; i<nMels+1; i++ ) {  fdiff[i] = freqs[i+1]-freqs[i]; }
        // Get the weights;
        float[][] weights = new float[nMels][nFreqs];
        for (int m=0; m < nMels; m++ ) {
            for (int k=0; k < nFreqs; k++) {
                weights[m][k] = (float) Math.max(
                        0, Math.min((-1.0)*ramps[m][k]/fdiff[m],ramps[m+2][k]/fdiff[m+1]));}}
        // Slaney-style mel is scaled to be approx constant energy per channel;
        float[][] norm_weights = new float[nMels][nFreqs];
        for (int i=0; i< nMels; i++) {
            for (int j=0; j<nFreqs; j++ ) {
                norm_weights[i][j] = weights[i][j] * (2.0f / (freqs[i+2] - freqs[i]));}}
        return norm_weights;
    }

    private float[][] power_to_db(float[][] S) {
        float[][] magnitude = new float[S.length][S[0].length];
        // Calculate Magnitude and Reference:
        float ref = S[0][0];
        for (int i=0; i< S.length; i++) {
            for (int j=0; j< S[0].length; j++) {
                magnitude[i][j] = (float)(10.0f * Math.log10(Math.max(1e-10, S[i][j])));
                ref = Math.max(ref,S[i][j]);}}
        // Convert Log:
        ref = (float) (10.0f * Math.log10(Math.max(1e-10, ref)));
        float magmax = magnitude[0][0] - ref;
        for (int i=0; i< S.length; i++) {
            for (int j=0; j< S[0].length; j++) {
                magnitude[i][j] = magnitude[i][j] - ref;
                magmax = Math.max(magmax, magnitude[i][j]);}}
        // Get Spectrogram in dB
        for (int i=0; i< S.length; i++) {
            for (int j=0; j< S[0].length; j++) {
                magnitude[i][j] = (float)(Math.max(magnitude[i][j], (magmax - 80.0)));}}
        return magnitude;
    }

    private float[][] flipUD(float[][] spec){
        float[][] spectrogramFlipped = new float[nMels][frames];
        for (int i=0; i< nMels; i++) {
                System.arraycopy(spec[(nMels-1-i)], 0, spectrogramFlipped[i], 0, frames);
        }
        return spectrogramFlipped;
    }

    private float[][] normalize(float[][] spec){
        float[][] normalizedSpectrogram = new float[nMels][frames];
        for (int j=0 ; j < frames ; j++ ) {
            for (int i = 0; i < nMels; i++) {
                normalizedSpectrogram[i][j] = (spec[i][j]+80)/80;} }
        return normalizedSpectrogram;
    }
} // class end