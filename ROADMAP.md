# Roadmap

1. ~~retrain with 1024x768: small errors in 512x384 would become huge errors in 6016x4512. 1 pixel in 512x384 is 0.195% but a 19% error is 11.4 pixels!~~
2. ~~No normalization in `compute_coords_from_segmentation_mask`, store as uint8. Geometric transforms work more efficiently on uint8!~~
3. ~~Add to transforms: RandomPerspective, ScaleJitter, RandomRotation, ElasticTransform, RandomChannelPermutation, RandomResize,  GaussianBlur, GaussianNoise, RandomAffine. **Maybe exclude RandomResize**~~
4. ~~Balance CPU and GPU transforms: CPU is sleeping with JPEG only but all of them is too much.~~
5. Resume training
6. read exif (PIL) or check w > h to understand if the image I'm doing predictions on is rotated (horizontal)
7. **shrink the corners** before storing them in the dataabase: if someone asked me to crop an image of a document, I WOULD NOT crop it EXACTLY at the borders of the document, I'd sacrifice a little bit of page to 1. exclude page/border imperfections that would reveal the background and 2. because this way, even if the model makes a mistake, there's a higher chance that it will not output coordinates of corners outside the borders of the document
8. increment the dataset with my own training examples, the dataset borders are way too perfect
9. Normalize corners to allow for different sizes of pictures (same ratio though)
10. Write a walkthrough
