https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_Efficient_Classification_of_Very_Large_Images_With_Tiny_Objects_CVPR_2022_paper.pdf

INFO:search-accumulate-grad:best_params: {'lr': 0.00038900971610501784, 'accumulation_steps': 1} best_value: 0.16393442451953888 image_size:1024 batch_size:15 ratio: 0.15
Trial 6 finished with value: 0.1886792629957199 and parameters: {'lr': 0.00031655209394667395, 'ratio': 0.15069802469010238, 'accumulation_steps': 2} image_size: 512 batch_size:16

-> lr: 0.00031655209394667395 ratio: 0.15069802469010238 accumulation_steps: 2 image_size:1024 batch_size:16
-> lr: 0.00031655209394667395 * 2 / 3 ratio: 0.15069802469010238 accumulation_steps: 3 image_size:1024 batch_size:16 / 3 * 2 ~ 10
-> trial-0.yaml

Trial 9 finished with value: 0.1904761791229248 and parameters: {'lr': 0.0005094480973201417, 'ratio': 0.5369031081273525, 'accumulation_steps': 2}. Best is trial 9 with value: 0.1904761791229248. image_size: 512 batch_size:16

-> lr: 0.0005094480973201417, ratio: 0.5369031081273525 accumulation_steps: 2 image_size:1024 batch_size:15 image_size:15
-> trial-1.yaml

#
解像度512で max score 0.1656
model: b4
ratio: 0.275
lr: 0.00054
scale: 0.2
transtale: 0.01 -> 0?
vflip: 0.5
affine_p: 0.5
use_roi: false
accumulation_steps:2
