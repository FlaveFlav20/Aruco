## Example:

```sh
mkdir out_calib
python3 calibration.py -v=0 -X=5 -Y=5 -S=0.05 -o=out_calib
```

## Explain

- **v** : video source -> You can use it to change used camera (default 0)
- **X** : Numbers of horizontal chess cases (madatory)
- **Y** : Numbers of vertical chess cases (mandatory)
- **S** : A square size in meters
- **o** : Path destination of calibration file. The name file will be calibration_params.yml
