# Roadmap

1. ~~retrain with 1024x768: small errors in 512x384 would become huge errors in 6016x4512. 1 pixel in 512x384 is 0.195% but a 19% error is 11.4 pixels!~~
2. Add to transforms: RandomPerspective, ColorJitter, GaussianBlur
3. read exif (PIL) or check w > h to understand if the image I'm doing predictions on is rotated (horizontal)
4. **shrink the corners** before storing them in the dataabase: if someone asked me to crop an image of a document, I WOULD NOT crop it EXACTLY at the borders of the document, I'd sacrifice a little bit of page to 1. exclude page/border imperfections that would reveal the background and 2. because this way, even if the model makes a mistake, there's a higher chance that it will not output coordinates of corners outside the borders of the document
5. increment the dataset with my own training examples, the dataset borders are way too perfect
