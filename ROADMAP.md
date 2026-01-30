# Roadmap

1. ~~retrain with 1024x768: small errors in 512x384 would become huge errors in 6016x4512. 1 pixel in 512x384 is 0.195% but a 19% error is 11.4 pixels!~~
2. ~~No normalization in `compute_coords_from_segmentation_mask`, store as uint8. Geometric transforms work more efficiently on uint8!~~
3. ~~Add to transforms: RandomPerspective, ScaleJitter, RandomRotation, ElasticTransform, RandomChannelPermutation, RandomResize,  GaussianBlur, GaussianNoise, RandomAffine. **Maybe exclude RandomResize**~~
4. ~~Balance CPU and GPU transforms: CPU is sleeping with JPEG only but all of them is too much.~~
5. Make each dataset source provide a lambda to `crawl` to generate columns in the crawl output
6. Resume training
7. read exif (PIL) or check w > h to understand if the image I'm doing predictions on is rotated (horizontal)
8. **shrink the corners** before storing them in the dataabase: if someone asked me to crop an image of a document, I WOULD NOT crop it EXACTLY at the borders of the document, I'd sacrifice a little bit of page to 1. exclude page/border imperfections that would reveal the background and 2. because this way, even if the model makes a mistake, there's a higher chance that it will not output coordinates of corners outside the borders of the document
9. increment the dataset with my own training examples, the dataset borders are way too perfect
10. Normalize corners to allow for different sizes of pictures (same ratio though)
11. Write a walkthrough
12. handle several ratios (4:3, 16:9)
13. Tensorboard: display debug images
14. The number of parameters for training has become too high. Change approach and use a YAML config file (no more cli). In the code use a config factory: `class ConfigSource(Enum)`:
    1. variant 1: YAML(path or env variable)
    2. variant 2: AWS_PARAMETER_STORE(string_to_parse or env variable to get info)
    3. variant 3: other...
    
    and then (pydantic?) ConfigSource.from_str(s). If is isPath and Path.exists() parse YAML
15. ~~Make crawk multithreaded~~
16. ~~Store original coordinates, only compute recess at runtime! => the user already has the option to define --recess (default=0)~~