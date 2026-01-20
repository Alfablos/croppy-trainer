https://lmdb.readthedocs.io/

python main.py crawl --data-root /home/antonio/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --iext '_in.png' --lext='_gt.png' -o ./dataset_float32.csv -v -n -c --precision f32

python main.py pc --data-map ./dataset_float32.csv -o training_data --height 512 --width 384 --compute-corners --strict --precision f32 --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet -v

# or, alltogether
python main.py pc -o training_data --height 512 --width 384 --compute-corners --strict --precision f32 --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train -v