package com.audio.sceneclassifier;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuffXfermode;
import android.graphics.PorterDuff.Mode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.view.View;

@SuppressLint("ViewConstructor")
public class SpectrogramView extends View {
    private Paint paint = new Paint();
    private Bitmap rawBitmap;
    private Bitmap spectrogramBitmap;
    private int scale;
    private int round;
    private int width;
    private int height;

    SpectrogramView(Context context,double[][] data, String colorMap,int scale,int round) {
        super(context);
        this.scale = scale;
        this.round = round;
        if (data != null) {
            paint.setStrokeWidth(1);
            this.width = data.length;
            this.height = data[0].length;
            int[] colors = new int[width*height];
            double minValue = 1;
            double maxValue = 0;
            int pixel = 0;
            for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                    if (data[j][i] > maxValue) {maxValue =data[j][i];}
                    if (data[j][i] < minValue) {minValue =data[j][i];}
                    int[] rgbValue = getRGB(data[j][i],colorMap);
                    colors[pixel] = Color.argb(rgbValue[0],rgbValue[1],rgbValue[2],rgbValue[3]);
                    pixel ++;}}
            rawBitmap = Bitmap.createBitmap(colors, width, height, Bitmap.Config.ARGB_8888);
            spectrogramBitmap = roundedCorners(rawBitmap);
            spectrogramBitmap = scaleBitmap(spectrogramBitmap);
        } else { System.err.println("Data Corrupt");}
    } // SpectrogramView Constructor end
        
   public Bitmap getBitmap(){
        return spectrogramBitmap;
    } // getSpectrogramView method end
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(rawBitmap, 0, 0, paint); // top=100
    } // onDraw override end

    private Bitmap roundedCorners(Bitmap inputBitmap) {
        Bitmap outputBitmap = Bitmap.createBitmap(
                inputBitmap.getWidth(), inputBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(outputBitmap);
        final int color = 0xff424242;
        final Paint paint = new Paint();
        final Rect rect = new Rect(0, 0, inputBitmap.getWidth(), inputBitmap.getHeight());
        final RectF rectF = new RectF(rect);
        paint.setAntiAlias(true);
        canvas.drawARGB(0, 0, 0, 0);
        paint.setColor(color);
        canvas.drawRoundRect(rectF, round, round, paint);
        paint.setXfermode(new PorterDuffXfermode(Mode.SRC_IN));
        canvas.drawBitmap(inputBitmap, rect, rect, paint);
        return outputBitmap;
    } // getRoundedCornerBitmap method end

    private Bitmap scaleBitmap(Bitmap inputBitmap) {
        return Bitmap.createScaledBitmap(
                inputBitmap, width * scale, height * scale, true);
    }

    private int[] getRGB(double inputValue,String colorMap){
        // get the Magma RGB values for a double input with range [0,1] 
        int[] result = new int[4]; // (alpha, red, green, blue)
        double[] reds;
        double[] blues;
        double[] greens;

        // DEFAULT COLORMAP: MAGMA
        reds = new double[] {0,0,28,79,129,181,229,251,254,251,255};
        greens = new double[] {0,0,16,18,37,54,89,135,194,253,255};
        blues = new double[] {0,4,68,123,129,122,100,97,135,191,255};
        switch (colorMap) {
            case "MAGMA":
                reds = new double[]{0, 0, 28, 79, 129, 181, 229, 251, 254, 251, 255};
                greens = new double[]{0, 0, 16, 18, 37, 54, 89, 135, 194, 253, 255};
                blues = new double[]{0, 4, 68, 123, 129, 122, 100, 97, 135, 191, 255};
                break;
            case "PLASMA":
                reds = new double[]{4, 12, 75, 125, 168, 203, 229, 248, 253, 240, 246};
                greens = new double[]{0, 8, 3, 3, 34, 70, 107, 148, 195, 249, 255};
                blues = new double[]{131, 135, 161, 168, 150, 121, 93, 65, 40, 33, 39};
                break;
            case "VIRIDIS":
                reds = new double[]{67, 68, 71, 59, 44, 33, 39, 92, 170, 253, 255};
                greens = new double[]{0, 1, 44, 81, 113, 114, 173, 200, 220, 231, 233};
                blues = new double[]{83, 84, 122, 139, 142, 141, 129, 99, 50, 37, 39};
                break;
        }
        int start = 0;
        int end = 0;
        double scaledValue;
        result[0] = 255; // set alpha to max.
        // Set ColorMap Indexes from data.
        if (inputValue<0.1){ end = 1;}
        else if (inputValue<0.2){ start = 1;end = 2;}
        else if (inputValue<0.3){ start = 2;end = 3;}
        else if (inputValue<0.4){ start = 3;end = 4;}
        else if (inputValue<0.5){ start = 4;end = 5;}
        else if (inputValue<0.6){ start = 5;end = 6;}
        else if (inputValue<0.7){ start = 6;end = 7;}
        else if (inputValue<0.8){ start = 7;end = 8;}
        else if (inputValue<0.9){ start = 8;end = 9;}
        else if (inputValue<=1.0){ start = 9;end = 10;}
        // get color index from gradient interpolations
        scaledValue = (10*inputValue)-start;
        if (scaledValue < 0.0) { scaledValue = 0; }
        if (scaledValue > 1.0) { scaledValue = 1; }
        result[1] = (int) ((scaledValue) * reds[end] - (scaledValue-1.0) * reds[start]);
        result[2] = (int) ((scaledValue) * greens[end] - (scaledValue-1.0) * greens[start]);
        result[3] = (int) ((scaledValue) * blues[end] - (scaledValue-1.0) * blues[start]);
        // upper & lower safety limits
        if (result[1] < 0) {result[1] = 0; }
        if (result[2] < 0) {result[2] = 0; }
        if (result[3] < 0) {result[3] = 0; }
        if (result[1] > 255) {result[1] = 255; }
        if (result[2] > 255) {result[2] = 255; }
        if (result[3] > 255) {result[3] = 255; }
        return result;
    } // getMagmaRGB method end
} // class end