Imaged loaded Retina Image with size = 584×565 

1. retina255.tif: 
Each pixel / 255 #why?

2. respBeforeRescale.tif:
The response image is rescaled by multiplying by 0.5 (preprocessthresh) and then normalizing
see rescaleImage(resp .* mask, 0, 255) function

3. respimage.tif: 
The filter response

4. resp_imagesc.tif:
How matlab displays the image (data is scaled to use the full colormap.)

5. segmented.tif:
The respimage is converted to boolean (1 if > 37)

Reminder:
Before comparing the output, remember to use two filters (sym and asym) and add them [see media15.m]