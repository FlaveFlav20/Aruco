# Estimation_pos_CSV

## Example

```sh
python3 aruco_pos_csv.py -t=DICT_4X4_100 -o=/path_to_calibration/calibration.yml -l=0.08 -c=CSV/ -i=Image/ -b=0 -v=0 -n=1
```

## Explaination

python3 aruco_pos_csv.py -t=DICT_4X4_100 -o=/path_to_calibration/calibration.yml -l=0.08 -c=CSV/ -i=Image/ -b=0 -v=0 -n=1

- **t** : dictionary (default DICT_4X4_100)
- **o** : path to calibration file
- **l** : length of 1 marker
- **c** : destination folder of CSV result
- **i** : destination filder of captured images
- **b** : the begining ID of stating count
- **v** : video source
- **n** : number of markers expected

This program will create a CSV file. Structure of a line, if we put n=2 (2 markers was expected):
- number_capture,marker_id (dict =DICT_4X4_100),x,y,z,pitch,yaw,roll,marker_id (dict =DICT_4X4_100),x,y,z,pitch,yaw,roll

Each line get a unique ID (<=> -b (arguments)) and markers informations.
