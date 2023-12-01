# Text-detection models with obs
## Installation

```
# python packages
pip install -r requirements.txt
# pytesseract
apt install tesseract-ocr
```
## Run demo
```
bash run.sh
```
## Note
- The pytesseract is good at print detection.

- The dl models is good at handwriting detection.

## Param
- frame: frames per sec.

- use_dl: if use deep learning models. (or pytesseract)

- use_obs: if use obs input. (if not, you should assignation a image_folder)

- show: if show the camera.


## Citation
```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```