# neural-fonts - GAN을 활용한 한글 폰트 제작 프로젝트

<p align="center">
  <img src="assets/NanumBrush-gen15.png">
</p>

GAN을 사용하여 한글 폰트를 자동으로 만들어 주는 프로젝트입니다.

디자이너가 399자만 만들면 딥러닝을 통해서 해당 폰트의 style 정보를 훈련하여 11,172자의 완성형 한글을 생성합니다.

중국 폰트를 생성하는 [zi2zi](https://github.com/kaonashi-tyc/zi2zi)를 한글에 맞게 수정하여 사용하였습니다.

## 갤러리

### 필기체 (나눔 붓 폰트)

|             Original             |            Generated             |
| :------------------------------: | :------------------------------: |
| ![](assets/NanumBrush-org15.png) | ![](assets/NanumBrush-gen15.png) |

### 고딕체 (푸른전남 폰트)

|           Original           |          Generated           |
| :--------------------------: | :--------------------------: |
| ![](assets/Pureun-org15.png) | ![](assets/Pureun-gen15.png) |

## Overview

399자에 대해서만 훈련할 경우 완성형 한글의 글자 수에 비해서 입력 크기가 작습니다.

이를 해결하기 위해 먼저 다양한 한글 글꼴로 훈련한 모델을 만듭니다.

생성된 모델에 전이 학습을 통해 399자를 훈련하고 이를 사용하여 글꼴을 생성합니다.

## 사용법

### 요구 사항

- CUDA
- cudnn
- poetry

### 미리 훈련한 모델을 내려받거나 직접 훈련하기

먼저 한글 글꼴에 대해서 훈련한 모델을 생성합니다.

32개의 copyleft 글꼴에 대하여 미리 훈련한 모델을 내려받아 쓰거나 원하는 글꼴을 사용해 직접 훈련할 수도 있습니다.

> [미리 훈련한 모델 내려받기](https://mysnu-my.sharepoint.com/personal/yu65789_seoul_ac_kr/_layouts/15/guestaccess.aspx?docid=0a7fcfabb78af4958b790b98eccac135c&authkey=AVqeaI5jyQHWyklZgotc04Y) (링크 만료됨)

새로 훈련할 경우 [zi2zi](https://github.com/kaonashi-tyc/zi2zi)의 README를 참조하여 훈련하면 됩니다.

### 훈련 데이터 생성

먼저 [폰트 템플릿](template/TemplateKR.pdf)을 내려받아서 인쇄한 후, 칸에 맞춰 글씨를 씁니다.

다 채운 템플릿을 스캔한 후, 이미지 편집 도구(그림판 등)를 사용하여 여백 부분을 잘라냅니다.

잘라낸 이미지 파일의 이름을 페이지 순서대로 `1-uniform.png`, `2-uniform.png`, `3-uniform.png`로 변경합니다.

아래 명령어를 사용하여 글꼴 이미지를 생성합니다.

```sh
poetry run crop --src_dir=src_dir
               --dst_dir=dst_dir
```

`dst_dir`에 각 글자의 유니코드 값을 이름으로 하는 폰트 이미지 파일이 생성됩니다.

### 전처리

I/O 병목을 막기 위해서 전처리를 거쳐 바이너리를 생성한 다음 사용합니다.

아래 명령어를 실행하여 source 글꼴과 손글씨가 합쳐진 훈련용 글꼴 이미지를 생성합니다.

```sh
poetry run font2img --src_font=src.ttf
                   --dst_font=src.ttf
                   --sample_count=1000
                   --sample_dir=sample_dir
                   --label=0
                   --handwriting_dir=handwriting_dir
```

`sample_dir`은 훈련을 위한 글꼴 이미지를 저장할 폴더입니다.
`handwriting_dir` 옵션을 사용해 템플릿으로 생성한 폰트 이미지 폴더를 알려줍니다.
`label` 옵션은 category embedding에서의 font index를 나타내며 기본값은 0입니다. 여러개의 폰트를 훈련하고 싶은 경우 각각의 폰트에 다른 `label`을 할당하면 됩니다.

이미지 생성이 완료되면 **package.py**를 실행해 이미지를 묶어 바이너리 형식으로 만듭니다.

```sh
poetry run package --fixed_sample=1
                  --dir=image_directory
                  --save_dir=binary_save_directory
```

명령어를 실행하면 **train.obj**가 `save_dir`에 생성됩니다. 해당 파일이 training을 위해 사용되는 data입니다.

### 연구 레이아웃

```sh
experiment/
└── data
    └── train.obj
```

루트 폴더 밑에 폰트를 위한 폴더를 생성한 다음 앞에서 생성한 바이너리 파일을 `data` 폴더 밑으로 옮깁니다.

### 훈련

훈련은 두 단계로 진행되는데, 1차 훈련 단계가 진행되고, 그 다음으로 파인 튜닝 단계가 진행됩니다.

아래의 명령어를 실행하여 훈련을 수행합니다.

#### 1단계

```sh
poetry run train --experiment_dir=experiment 
                --experiment_id=0
                --batch_size=16 
                --lr=0.001
                --epoch=30 
                --sample_steps=100 
                --schedule=10 
                --L1_penalty=100 
                --Lconst_penalty=15
```

#### 2단을

```sh
poetry run train --experiment_dir=experiment 
                --experiment_id=0
                --batch_size=16 
                --lr=0.001
                --epoch=120 
                --sample_steps=100 
                --schedule=40 
                --L1_penalty=500 
                --Lconst_penalty=1000
```

### 추론

훈련이 끝난 후 아래 명령어를 통해 추론을 수행합니다.

```sh
poetry run infer --model_dir=checkpoint_dir/ 
                --batch_size=16 
                --source_obj=binary_obj_path 
                --embedding_ids=label[s] of the font, separate by comma
                --save_dir=save_dir/
```

## 감사의 말

- [kaonashi-tyc](https://github.com/kaonashi-tyc)의 [zi2zi](https://github.com/kaonashi-tyc/zi2zi) 저장소 코드를 재구성해 파생시켰습니다.
- [periannath](https://github.com/periannath)의 [neural-fonts](https://github.com/periannath/neural-fonts) 저장소 코드를 바탕으로 수정했습니다.

## 라이선스

[MIT 라이선스](LICENSE)
