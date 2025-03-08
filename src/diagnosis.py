
import os, utils
import pandas as pd
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
from keras import backend as K
import soundfile, tempfile
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import configure

logger = configure.getLogger('diagnosis')
now=datetime.now()
models = {
    'q3': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q3/MCI_AD_q3.h5"),
            'NOR_AD': load_model("models/final_model/q3/SCI_AD_q3.h5"),
            'NOR_AB': load_model("models/final_model/q3/normal_abnormal_q3.h5"),
            'NOR_MCI': load_model("models/final_model/q3/SCI_MCI_q3.h5")
        },
        'seq': [8, 3, 5]
    },
    'q10': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q10/MCI_AD_ver1_q10.h5"),
            'NOR_AD': load_model("models/final_model/q10/normal_AD_ver1_q10.h5"),
            'NOR_AB': load_model("models/final_model/q10/normal_abnormal_ver1_q10.h5"),
            'NOR_MCI': load_model("models/final_model/q10/normal_MCI_ver1_q10.h5")
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    },
    'q11': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q11/MCI_AD.h5"),
            'NOR_AD': load_model("models/final_model/q11/normal_AD.h5"),
            'NOR_AB': load_model("models/final_model/q11/normal_abnormal.h5"),
            'NOR_MCI': load_model("models/final_model/q11/normal_MCI.h5")
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    'MCI_AD': [load_model(os.path.join('models', 'MA', item)) for item in sorted(os.listdir('models/MA'))],
    'NOR_AD':  [load_model(os.path.join('models', 'NA', item)) for item in sorted(os.listdir('models/NA'))],
    'NOR_AB':  [load_model(os.path.join('models', 'NAB', item)) for item in sorted(os.listdir('models/NAB'))],
    'NOR_MCI':  [load_model(os.path.join('models', 'NM', item)) for item in sorted(os.listdir('models/NM'))],
}

# voice_cut2 기준(최종 결정 완료)에 따른 문항별 음원 지속 시간
end_time = [9, 10, 12, 18, 60, 60, 60, 60, 60, 60, 60]
# ex) 1번 문항은 9초로 통일, 3번 문항은 12초로 통일..
sr = 48000 # sampling rate


def imageprocessing(wav_file, save_file, sr):
    Mel_Spectrogram(wav_file, save_file, sr)
    crop_mel(save_file)
  
    
def execute(fileList: list):
    '''
    fileList는 리스트 형태이며, 각 요소는 녹음파일의  configure.recordFileRoot 기준 full path 문자열.
    따라서 첫번째 녹음파일의 full path는 configure.recordFileRoot + fileList[0]
    configure.recordFileRoot값은 configure.json에서 recordFileRoot키의 값을 세팅하면됨.
    예)
        configure.recordFileRoot = '/data'
        fileList[0] = '/record/2020/02/25/322341234_DFEF123'
        일경우 실제 파일 full path는 /data/record/2020/02/25/322341234_DFEF123


    tempfile 사용예
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
        K.models.save_model(model.model, fd.name, overwrite=True)
        keras_model = fd.read()

    tempfile.gettempdir()
    tempfile.mktemp(suffix, prefix, dir)

    결과 샘플
    {
        "dementiaClass": "정상",
        "dementiaProba": 0.9
    }

    {
        "dementiaClass": "MCI",
        "dementiaClass": 0.7
    }
    '''
    try:
        result_proba_dict = dict()
        result_proba_list_dict = dict()       
        for model_class in models["q11"]['final_model'].keys(): #4가지 유형 불러옴
            img_data_array = {}
            img_columns_array = {}
            model_class1, model_class2 = model_class.split('_')


    # 여기서 수정해야함.모델이 11개 로드되는 포문이므로 10번문제가 빠졌을때 11번 문제가 10번 모델에 적용됨.
            for step_no, seq in enumerate(models['q{}'.format(len(fileList))]['seq']):
                loaded_model = models[model_class][seq - 1]
                
                mp4_file = os.path.join(configure.recordFileRoot,fileList[step_no])
                
                y, _ = librosa.load(mp4_file, sr=48000)
                
                duration = len(y)/48000

                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=True).name
                if (end_time[step_no] > duration):
                    final_duration=end_time[step_no]*sr
                    y2 = np.concatenate((y, np.zeros(abs(final_duration-len(y)))), axis=0)


                else:
                    cutDuration = sr * end_time[step_no]
                    y2 = y[:cutDuration]
                


                soundfile.write(temp_file, y2, sr, format='WAV')

                imageprocessing(temp_file, "{}.png".format(mp4_file), sr)
                        

                img_name = os.path.join(configure.recordFileRoot, "{}.png".format(fileList[step_no]))
                feature_set, features_column,  = feature_extract(loaded_model, seq, img_name)
                
                K.clear_session()
                #디버그 툴로 꼭 확인할것 문제 제거시 실수 확율 매우 높음)
                img_data_array["data" + str(seq-1)] = feature_set
                img_columns_array["col" + str(seq-1)] = features_column

            final_df = final_classification(img_data_array, img_columns_array, ques_no= len(fileList))

            dementia_proba2 = models["q{}".format(len(fileList))]['final_model'][model_class].predict(final_df)[0,0]
            dementia_proba1 = 1 - dementia_proba2
            result_proba_dict[model_class] = int(dementia_proba1*100000)/1000
            
            
            model_class1_proba = result_proba_list_dict.pop(model_class1, [])
            model_class1_proba.append(dementia_proba1)
            result_proba_list_dict[model_class1] = model_class1_proba
            
            model_class2_proba = result_proba_list_dict.pop(model_class2, [])
            model_class2_proba.append(dementia_proba2)
            result_proba_list_dict[model_class2] = model_class2_proba
            
        ab_proba = result_proba_list_dict.pop('AB')[0]
        # abnormal 부분 반으로 점수 줬으나 실험결과 MCI는 원점수를 주는것이 선별 정확도가 더 높아 /2를 제거
        # 이로인해 합계가 100이 안되는 상황 발생.
        result_proba_list_dict['MCI'].append(ab_proba/2) #abnormal 부분 점수 반으로 나눔
        result_proba_list_dict['AD'].append(ab_proba/2) #abnormal 부분 점수 반으로 나눔

        sum_dict = {k: sum(v) for k, v in result_proba_list_dict.items()}

        individualScore={"classScore":result_proba_dict, "sum_dict":{k:int(v*1000/3)/1000 for k,v in sum_dict.items()}}
           
        #"AD"와 "MCI"를 더한값과 normal을 비교하게 되서 아래 MAX로 dementia_class를 정하는 방식은 주석처리
        dementia_class = max(sum_dict, key=sum_dict.get)
        #식약처 normal vs abnormal 표현으로 바꾸기 위해서 dementia_proba 변경
        dementia_proba = sum_dict[dementia_class] / len(result_proba_list_dict[dementia_class])
        individualScore['proba'] = int(sum_dict[dementia_class]*1000 / len(result_proba_list_dict[dementia_class])) / 10

        results = dict(
            dementiaClass=dementia_class,
            dementiaProba=int(dementia_proba*1000)/10,
            individualScore=utils.json_dumps(individualScore),
            )

        logger.debug('{}'.format(results))

        return results
        
    except Exception as ex:
        logger.exception('')
        result = {"resultCode": "9999", "msg": '{}: {}'.format(type(ex).__name__, str(ex))}
        
        return result
    

def feature_extract(loaded_model, step_num, img_path):
    flatten = loaded_model.layers[-2].output  # 원하는 레이어를 자동으로 가져옴
    test_model = Model(inputs=loaded_model.input, outputs=flatten)
    img = image.load_img(img_path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0  # Normalize

    # Extract features
    feature_set = test_model.predict(img_tensor)
    features_column = [f"img_f{i}_{step_num}" for i in range(feature_set.shape[1])]

    return feature_set, features_column


def final_classification(img_data, img_column, ques_no=11):
    # 데이터 구조화
    #img_data_dict = {f"data{step_num}": img_data[step_num] for step_num in range(len(img_data))}
    #img_column_dict = {f"col{step_num}": img_column[step_num] for step_num in range(len(img_column))}

    # 문제별 매핑
    SEQUENCE_MAPPING = {
        11: ["data7", "data2", "data0", "data6", "data4", "data5", "data10", "data3", "data9", "data1", "data8"],
        10: ["data7", "data2", "data0", "data6", "data4", "data5", "data10", "data3", "data1", "data8"],
        3:  ["data7", "data2", "data4"],
    }

    # 데이터 처리
    if ques_no in SEQUENCE_MAPPING:
        selected_data_keys = SEQUENCE_MAPPING[ques_no]
        data_all, columns_all = process_data(img_data, img_column, selected_data_keys)
    else:
        raise ValueError(f"Unsupported question number: {ques_no}")

    # 데이터프레임 생성
    df = pd.DataFrame(data=data_all,columns=columns_all)
    return df

# 잘린 파일을 mel-spectrogram으로 변환하기
def Mel_Spectrogram(wav_path, save_file, sr):
    y, sr = librosa.load(wav_path, sr=sr)

    sec = 60

    index = sr * sec

    y_segment = y[0:index]

    S = librosa.feature.melspectrogram(y=y_segment, sr=sr, win_length=3000,
                                       n_fft=3000)  # win_length, n_fft = 3000으로 통일.(<-가천대에서 결정)
    sum_s=np.sum(S)
    plt.gcf().set_size_inches(3, 3.24)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.savefig(save_file, dpi=100)
    plt.close()

def crop_mel(img_file):
    img = plt.imread(img_file)
    img_cropped = img[24:324, :, :]

    plt.gcf().set_size_inches(3, 3)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.imshow(img_cropped)

    plt.savefig(img_file, dpi=100)
    plt.close()

def process_data(img_data_dict, img_column_dict, keys):
    """
    Helper function to process data and columns.
    """
    data = np.hstack([img_data_dict[key] for key in keys])
    columns = np.hstack([img_column_dict[key.replace("data", "col")] for key in keys])
    return data, columns
if __name__ == '__main__':
        
    #pass
    
    # final=execute(["/0b233392-7574-4b16-bb25-45f5e5dcb8dd_0_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_1_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_2_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_3_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_4_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_5_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_6_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_7_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_8_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_9_R",
    #                "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_10_R"
    #                ])
    final=execute(["fa352139-508c-40bf-a49d-db57de4f4a0b_0.wav","fa352139-508c-40bf-a49d-db57de4f4a0b_1.wav",
                   "fa352139-508c-40bf-a49d-db57de4f4a0b_2.wav"])


    # final=execute(["/1694758181080_o3uyZ4",
    #                "/1694758250760_vvW2lP",
    #                "/1694758342430_yPKM6v",        
    #                ])