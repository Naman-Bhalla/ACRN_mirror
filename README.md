The code here doesn't belong to me. Instead, it is a mirror for https://sigir2018.wixsite.com/acrn 's ACRM Code, which is originally present on Baidu at: https://pan.baidu.com/s/1eUgvASi .

The original code also contains the following files at the root directory: 
- Interval128_256_overlap0.8_c3d_fc6.tar : https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view
- TACoS.tar : https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view
- Interval64_128_256_512_overlap0.8_c3d_fc6.tar : https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view

The above Google Drive URLs don't belong to me either and are taken from https://github.com/jiyanggao/TALL .

The structure of your directory should be:
```
- ACRN
    - util
        - __init__.pyc
        - __init__(1).pyc
        - cnn.pyc
        - cnn.py
        - cnn(1).pyc
        - cnn(1).py
    - exp_data
        - TACoS
            - TACoS_val_videos.txt
            - TACoS_val_samples.txt
            - TACoS_train_videos.txt
            - TACoS_train_samples.txt
            - TACoS_test_videos.txt
            - TACoS_test_samples.txt

    - ACRN_model.py
    - dataset.py
    - readme.docx
    - main.py
    - Interval64_128_256_512_overlap0.8_c3d_fc6.tar
    - Interval128_256_overlap0.8_c3d_fc6.tar
    - TACoS.tar
    - video_allframes_info.pkl
    - vs_multilayer.py
    - vs_multilayer.pyc
```

```
@inproceedings{Liu:2018:AMR:3209978.3210003,
 author = {Liu, Meng and Wang, Xiang and Nie, Liqiang and He, Xiangnan and Chen, Baoquan and Chua, Tat-Seng},
 title = {Attentive Moment Retrieval in Videos},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {15--24},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3210003},
 doi = {10.1145/3209978.3210003},
 acmid = {3210003},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {cross-modal retrieval, moment localization, temporal memory attention, tensor fusion},
} 
```