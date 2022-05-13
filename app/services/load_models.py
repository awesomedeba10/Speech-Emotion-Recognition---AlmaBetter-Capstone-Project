import pickle, os
import numpy as np
from typing import Union
import librosa
from app import app
from tensorflow import keras

class LoadModel:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_base_dir = os.path.join(app.config.get('BASE_DIR'), 'models')
        self.standard_scaler = 'standard_scaler.pkl'
        self.mlp_model = 'mlp_model_256_100_100.pkl'
        self.lbl_encoder = 'lbl_encoder.pkl'
        self.onehot_encoder = 'onehot_encoder.pkl'
        self.conv1d_model = 'conv_model_256_128_64.h5'

    def predict(self, file) -> str:
        if self.model_name == 'mlp':
            lbl = self.__predict_mlp(file)
            lbl = lbl[0]
        elif self.model_name == 'conv1d':
            lbl = self.__predict_conv1d(file)
            lbl = lbl[0][0]

        return lbl


    def __load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __predict_mlp(self, audio_path):
        model = self.__load_pickle(os.path.join(self.model_base_dir, self.mlp_model))
        X_feature = self.__run_feature_extraction(audio_path)

        scaler = self.__load_pickle(os.path.join(self.model_base_dir, self.standard_scaler))
        X_feature = scaler.transform(X_feature)

        pred = model.predict(X_feature)

        lblencoder = self.__load_pickle(os.path.join(self.model_base_dir, self.lbl_encoder))
        pred_lbl = lblencoder.inverse_transform(pred)
        return pred_lbl

    def __predict_conv1d(self, audio_path):
        model = keras.models.load_model(os.path.join(self.model_base_dir, self.conv1d_model))
        X_feature = self.__run_feature_extraction(audio_path)

        scaler = self.__load_pickle(os.path.join(self.model_base_dir, self.standard_scaler))
        X_feature = scaler.transform(X_feature)
        X_feature = np.expand_dims(X_feature, axis=2)

        pred = model.predict(X_feature)

        onehotencoder = self.__load_pickle(os.path.join(self.model_base_dir, self.onehot_encoder))
        pred_lbl = onehotencoder.inverse_transform(pred)
        return pred_lbl

    def __run_feature_extraction(self, audio_path: str) -> np.ndarray:
        """Run feature extraction process on given dataframe

        Args:
            data (_type_): pandas dataframe to extract features from
            feature_col (Union[list, str]): feature column name or list

        Returns:
            np.ndarray: numpy array with all features and augmentated data
        """
        file_path_arr = np.array([audio_path]).reshape(-1, 1)
        X = np.array(list(map(self.__get_feature, file_path_arr)))

        if X.ndim == 3:
            # creating 2d array ((i, j, k) -> ((i*j), k))
            X = X.reshape((X.shape[0]*X.shape[1]), X.shape[2])

        assert X.ndim == 2

        return X


    def __get_feature(self, path: Union[str, np.ndarray], add_noise: bool=False, add_stretch: bool=False) -> np.ndarray:
        """Add noise or stretch, pitch to the audio file and extract features from it

            Args:
            path (Union[str, np.ndarray]): paths to audio file or numpy array representing a audio file
            add_noise (bool, optional): add noise. Defaults to True.
            add_stretch (bool, optional): add stretching and pitching. Defaults to True.

        Returns:
            np.ndarray: numpy array with all features and augmentated data
        """
        if type(path).__module__ == np.__name__:
            path = path[0]

        file, sample_rate = librosa.load(path, offset=0.6, duration=2.5)

        result = np.array(self.__extract_feature(file, sample_rate=sample_rate))

        if add_noise:
            noise_arr = self.__extract_feature(self.__noise(file), sample_rate=sample_rate)
            result = np.vstack((result, noise_arr))

        if add_stretch:
            stretch_pitch_data = self.__pitch(self.__stretch(file), sampling_rate=sample_rate)
            stretch_pitch_arr = self.__extract_feature(stretch_pitch_data, sample_rate=sample_rate)
            result = np.vstack((result, stretch_pitch_arr))

        return result

    def __extract_feature(self, audio: np.ndarray, sample_rate: int, zcr: bool=True, chroma: bool=True,
        rms: bool=True, mfcc: bool=True, mel: bool=True) -> np.ndarray:
        """This function extract features from given audio file and return a numpy array
            with all features stacked in one row

        Args:
            audio (np.ndarray): numpy array representating a audio file
            sample_rate (int): sample rate of audio file
            zcr (bool, optional): extract zero crossing rate. Defaults to True.
            chroma (bool, optional): extract chroma. Defaults to True.
            rms (bool, optional): extract root mean square value. Defaults to True.
            mfcc (bool, optional): extract mfcc. Defaults to True.
            mel (bool, optional): extract mel. Defaults to True.

        Returns:
            np.ndarray: numpy array with all features stacked in one row
        """
        result = np.array([])
        if zcr:
            zcr_arr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
            result = np.hstack((result, zcr_arr))

        if chroma:
            stft = np.abs(librosa.stft(y=audio))
            chroma_arr = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_arr))

        if rms:
            rms_arr = np.mean(librosa.feature.rms(y=audio).T, axis=0)
            result = np.hstack((result, rms_arr))

        if mfcc:
            mfcc_arr = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mfcc_arr))

        if mel:
            mel_arr = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_arr))

        return result

    def __noise(self, data):
        """Add noise to audio data"""
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def __stretch(self, data, rate=0.8):
        """Stretch audio data"""
        return librosa.effects.time_stretch(data, rate=rate)

    def __shift(self, data):
        """Shift audio data"""
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def __pitch(self, data, sampling_rate, pitch_factor=0.7):
        """Pitch audio data"""
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)