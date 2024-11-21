# Estimation_pos_test

## Example

```sh
# /path_to_calibration/calibration.yml -> Calibration file
python3 aruco_pos_estimation.py -t=DICT_4X4_100 -o=/path_to_calibration/calibration.yml -l=0.08 -v=0
```

## Explaination

- **t** : ditionary dict (default DICT_4X4_100)
- **o** : calibration file destination
- **l** : length of 1 markers in meters (mandatory)
- **v** : video source (default 0)
